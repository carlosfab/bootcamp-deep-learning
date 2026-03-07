"""Modulo de dados LUNA16 - candidatos, CT scans e Dataset PyTorch."""

import copy
import csv
import functools
from collections import namedtuple
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "luna"


CandidateInfo = namedtuple(
    "CandidateInfo",
    "is_nodule, diameter_mm, series_uid, center_xyz",
)

IRC = namedtuple("IRC", "index, row, col")
XYZ = namedtuple("XYZ", "x, y, z")


def irc_to_xyz(coord_irc, origin_xyz, vx_size_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vx_size_a = np.array(vx_size_xyz)
    coord_xyz = (direction_a @ (cri_a * vx_size_a)) + origin_a
    return XYZ(*coord_xyz)


def xyz_to_irc(coord_xyz, origin_xyz, vx_size_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vx_size_a = np.array(vx_size_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vx_size_a
    cri_a = np.round(cri_a)
    return IRC(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))


@functools.lru_cache(1)
def load_candidates(require_on_disk=True):
    mhd_files = list(DATA_DIR.rglob("*.mhd"))
    present_on_disk = {p.stem for p in mhd_files}

    diameter_dict = {}
    with open(DATA_DIR / "annotations.csv") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            center_xyz = tuple(float(x) for x in row[1:4])
            diameter_mm = float(row[4])
            diameter_dict.setdefault(series_uid, []).append(
                (center_xyz, diameter_mm)
            )

    candidates = []
    with open(DATA_DIR / "candidates.csv") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if require_on_disk and series_uid not in present_on_disk:
                continue

            is_nodule = bool(int(row[4]))
            center_xyz = tuple(float(x) for x in row[1:4])

            candidate_diameter = 0.0
            for ann_xyz, ann_diam in diameter_dict.get(series_uid, []):
                for i in range(3):
                    if abs(center_xyz[i] - ann_xyz[i]) > ann_diam / 4:
                        break
                else:
                    candidate_diameter = ann_diam
                    break

            candidates.append(CandidateInfo(
                is_nodule, candidate_diameter, series_uid, center_xyz
            ))

    candidates.sort(reverse=True)
    return candidates


class CtScan:
    def __init__(self, series_uid):
        mhd_path = list(DATA_DIR.rglob(f"{series_uid}.mhd"))[0]
        ct_mhd = sitk.ReadImage(str(mhd_path))
        self.hu_a = np.array(
            sitk.GetArrayFromImage(ct_mhd), dtype=np.float32
        )
        self.hu_a.clip(-1000, 1000, self.hu_a)

        self.series_uid = series_uid
        self.origin_xyz = XYZ(*ct_mhd.GetOrigin())
        self.vx_size_xyz = XYZ(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def extract_crop(self, center_xyz, crop_size=(32, 48, 48)):
        center_irc = xyz_to_irc(
            center_xyz, self.origin_xyz,
            self.vx_size_xyz, self.direction_a,
        )
        slices = []
        for axis, center_val in enumerate(center_irc):
            start = int(round(center_val - crop_size[axis] / 2))
            end = int(start + crop_size[axis])
            if start < 0:
                start, end = 0, int(crop_size[axis])
            if end > self.hu_a.shape[axis]:
                end = self.hu_a.shape[axis]
                start = int(end - crop_size[axis])
            slices.append(slice(start, end))
        return self.hu_a[tuple(slices)], center_irc


@functools.lru_cache(1)
def get_ct(series_uid):
    return CtScan(series_uid)


class LunaDataset(Dataset):
    def __init__(self, val_stride=0, is_val=None, series_uid=None):
        self.candidates = copy.copy(load_candidates())
        if series_uid:
            self.candidates = [
                c for c in self.candidates
                if c.series_uid == series_uid
            ]
        if is_val:
            assert val_stride > 0
            self.candidates = self.candidates[::val_stride]
        elif val_stride > 0:
            del self.candidates[::val_stride]

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, idx):
        candidate = self.candidates[idx]
        ct = get_ct(candidate.series_uid)
        crop_a, center_irc = ct.extract_crop(candidate.center_xyz)

        crop_t = torch.from_numpy(crop_a).to(torch.float32)
        crop_t = crop_t.unsqueeze(0)
        label_t = torch.tensor(
            [not candidate.is_nodule, candidate.is_nodule],
            dtype=torch.long,
        )
        return crop_t, label_t, candidate.series_uid, torch.tensor(center_irc)

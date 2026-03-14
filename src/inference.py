"""Funcoes de inferencia para o LunaModel."""

from collections import defaultdict

import numpy as np
import torch

from luna_data import load_candidates, get_ct
from model import LunaModel


def load_model(checkpoint_path, device=None):
    """Carrega o LunaModel a partir de um checkpoint.

    Retorna (model, info_dict) onde info_dict contem
    epoch, best_f1 e history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = LunaModel()
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    info = {
        "epoch": ckpt["epoch"],
        "best_f1": ckpt["best_f1"],
        "history": ckpt.get("history"),
    }
    return model, info


def run_inference(candidates, model, device, batch_size=64,
                  max_cts=None, print_every=50):
    """Roda inferencia em uma lista de candidatos.

    Agrupa por series_uid pra carregar cada CT uma unica vez.
    Retorna dict com probs, labels, series_uids, center_xyzs.
    """
    uid_to_cands = defaultdict(list)
    for c in candidates:
        uid_to_cands[c.series_uid].append(c)

    uids = list(uid_to_cands.keys())
    if max_cts is not None:
        uids = uids[:max_cts]

    all_probs = []
    all_labels = []
    all_uids = []
    all_xyzs = []

    for i, uid in enumerate(uids):
        cands = uid_to_cands[uid]
        ct = get_ct(uid)

        crops = []
        labels = []
        xyzs = []
        for c in cands:
            crop_a, _ = ct.extract_crop(c.center_xyz)
            crops.append(crop_a)
            labels.append(int(c.is_nodule))
            xyzs.append(c.center_xyz)

        crops_t = torch.from_numpy(np.stack(crops)).float().unsqueeze(1)

        with torch.no_grad():
            for start in range(0, len(crops_t), batch_size):
                batch = crops_t[start:start + batch_size].to(device)
                _, probs = model(batch)
                all_probs.extend(probs[:, 1].cpu().numpy())

        all_labels.extend(labels)
        all_uids.extend([uid] * len(cands))
        all_xyzs.extend(xyzs)

        get_ct.cache_clear()

        if print_every > 0 and (i + 1) % print_every == 0:
            print(f"  {i + 1}/{len(uids)} CTs processados")

    return {
        "probs": np.array(all_probs),
        "labels": np.array(all_labels),
        "series_uids": all_uids,
        "center_xyzs": all_xyzs,
    }


def classify_ct(series_uid, model, device, batch_size=64):
    """Classifica todos os candidatos de um CT.

    Retorna lista de dicts ordenada por probabilidade (maior primeiro).
    Cada dict tem: series_uid, center_xyz, probability, is_nodule.
    """
    candidates = [
        c for c in load_candidates()
        if c.series_uid == series_uid
    ]
    if not candidates:
        return []

    ct = get_ct(series_uid)
    crops = []
    for c in candidates:
        crop_a, _ = ct.extract_crop(c.center_xyz)
        crops.append(crop_a)

    crops_t = torch.from_numpy(np.stack(crops)).float().unsqueeze(1)

    all_probs = []
    with torch.no_grad():
        for start in range(0, len(crops_t), batch_size):
            batch = crops_t[start:start + batch_size].to(device)
            _, probs = model(batch)
            all_probs.extend(probs[:, 1].cpu().numpy())

    get_ct.cache_clear()

    results = []
    for c, prob in zip(candidates, all_probs):
        results.append({
            "series_uid": c.series_uid,
            "center_xyz": c.center_xyz,
            "probability": float(prob),
            "is_nodule": c.is_nodule,
        })

    results.sort(key=lambda r: r["probability"], reverse=True)
    return results

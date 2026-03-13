"""Funcoes de treinamento e dataset balanceado para o LunaModel."""

import copy
import math
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from luna_data import load_candidates, get_ct


def augment_candidate(crop_t, augmentation_dict):
    """Aplica augmentation 3D a um crop de CT."""
    transform_t = torch.eye(4)

    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i, i] *= -1
        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            transform_t[i, 3] = offset_float * random_float
        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_t[i, i] *= 1.0 + scale_float * random_float

    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)
        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        transform_t @= rotation_t

    affine_t = F.affine_grid(
        transform_t[:3].unsqueeze(0).to(torch.float32),
        crop_t.unsqueeze(0).size(),
        align_corners=False,
    )
    augmented_t = F.grid_sample(
        crop_t.unsqueeze(0),
        affine_t,
        padding_mode='border',
        align_corners=False,
    ).squeeze(0)

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_t)
        noise_t *= augmentation_dict['noise']
        augmented_t += noise_t

    return augmented_t


class LunaTrainingDataset(Dataset):
    def __init__(self, val_stride=0, is_val=False,
                 ratio_int=0, augmentation_dict=None):
        candidates = copy.copy(load_candidates())
        if is_val:
            assert val_stride > 0
            candidates = candidates[::val_stride]
        elif val_stride > 0:
            del candidates[::val_stride]

        self.pos_list = [c for c in candidates if c.is_nodule]
        self.neg_list = [c for c in candidates if not c.is_nodule]
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict or {}
        self.shuffle_samples()

    def shuffle_samples(self):
        random.shuffle(self.pos_list)
        random.shuffle(self.neg_list)

    def __len__(self):
        if self.ratio_int:
            return 200_000
        return len(self.pos_list) + len(self.neg_list)

    def __getitem__(self, idx):
        if self.ratio_int:
            pos_idx = idx // (self.ratio_int + 1)
            if idx % (self.ratio_int + 1) == 0:
                pos_idx %= len(self.pos_list)
                candidate = self.pos_list[pos_idx]
            else:
                neg_idx = idx - 1 - pos_idx
                neg_idx %= len(self.neg_list)
                candidate = self.neg_list[neg_idx]
        else:
            all_list = self.pos_list + self.neg_list
            candidate = all_list[idx]

        ct = get_ct(candidate.series_uid)
        crop_a, center_irc = ct.extract_crop(candidate.center_xyz)
        crop_t = torch.from_numpy(crop_a).to(torch.float32)
        crop_t = crop_t.unsqueeze(0)

        if self.augmentation_dict:
            crop_t = augment_candidate(crop_t, self.augmentation_dict)

        label_t = torch.tensor(
            [not candidate.is_nodule, candidate.is_nodule],
            dtype=torch.long,
        )
        return crop_t, label_t, candidate.series_uid, torch.tensor(center_irc)


def compute_metrics(tp, fp, fn, tn):
    """Calcula accuracy, precision, recall e F1."""
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    acc_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1': f1,
        'acc_pos': acc_pos, 'acc_neg': acc_neg,
    }


def train_one_epoch(model, loader, optimizer, loss_fn, device, print_every=0):
    """Executa uma epoca de treinamento."""
    model.train()
    total_loss = 0.0
    tp = fp = fn = tn = 0
    n_batches = 0

    for batch in loader:
        crops, labels, _, _ = batch
        crops = crops.to(device)
        labels_idx = labels[:, 1].to(device)

        optimizer.zero_grad()
        logits, probs = model(crops)
        loss = loss_fn(logits, labels_idx)
        loss.backward()
        optimizer.step()

        preds = (probs[:, 1] > 0.5)
        actual = labels_idx.bool()
        tp += (preds & actual).sum().item()
        fp += (preds & ~actual).sum().item()
        fn += (~preds & actual).sum().item()
        tn += (~preds & ~actual).sum().item()

        total_loss += loss.item()
        n_batches += 1

        if print_every > 0 and n_batches % print_every == 0:
            avg = total_loss / n_batches
            print(f"  Batch {n_batches}/{len(loader)} | Loss: {avg:.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(tp, fp, fn, tn)
    metrics['loss'] = avg_loss
    return metrics


def validate(model, loader, loss_fn, device, print_every=0):
    """Executa validacao (sem gradientes)."""
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            crops, labels, _, _ = batch
            crops = crops.to(device)
            labels_idx = labels[:, 1].to(device)

            logits, probs = model(crops)
            loss = loss_fn(logits, labels_idx)

            preds = (probs[:, 1] > 0.5)
            actual = labels_idx.bool()
            tp += (preds & actual).sum().item()
            fp += (preds & ~actual).sum().item()
            fn += (~preds & actual).sum().item()
            tn += (~preds & ~actual).sum().item()

            total_loss += loss.item()
            n_batches += 1

            if print_every > 0 and n_batches % print_every == 0:
                avg = total_loss / n_batches
                print(f"  Batch {n_batches}/{len(loader)} | Loss: {avg:.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(tp, fp, fn, tn)
    metrics['loss'] = avg_loss
    return metrics

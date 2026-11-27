"""Dice and NSD per organ: multilabel prediction vs TotalSegmentator-dataset ground truth.

Ground truth layout (TotalSegmentator dataset): <subject>/segmentations/<organ>.nii.gz (binary masks).
Prediction: multilabel NIfTI with TotalSegmentator 'total' label ids (as written with ml=True).

Usage:
  python eval_seg.py --pred preds/s0011.nii.gz --gt-dir Totalsegmentator_dataset_v300/s0011 --out metrics.csv \
         --organs abdomen14 --class-map total_v3 [--tag fennecseg] [--nsd3]
"""
import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from labels import TASKS, ABDOMEN14 as ABDOMEN


def surface(mask):
    return mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1), border_value=0)


def nsd(gt, pr, spacing, tol_mm=2.0):
    """Normalised surface distance (surface Dice) at tolerance tol_mm."""
    sg, sp = surface(gt), surface(pr)
    if not sg.any() and not sp.any():
        return np.nan
    if not sg.any() or not sp.any():
        return 0.0
    dt_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    ok = (dt_g[sp] <= tol_mm).sum() + (dt_p[sg] <= tol_mm).sum()
    return ok / (sg.sum() + sp.sum())


def bbox_crop(*masks, margin=3):
    """Crop to the union bounding box of all masks to speed up distance transforms."""
    union = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        union |= m
    if not union.any():
        return masks
    idx = np.argwhere(union)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = idx.max(0) + margin + 1
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    return tuple(m[sl] for m in masks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--organs", default="abdomen", help="'abdomen', 'all', a group name (labels.py) or comma-separated names")
    ap.add_argument("--tag", default="")
    ap.add_argument("--class-map", default="total", help="'total' or 'total_v3'")
    ap.add_argument("--nsd3", action="store_true", help="also NSD at 3 mm (column nsd3mm), TotalSegmentator's tolerance")
    args = ap.parse_args()

    from totalsegmentator.map_to_binary import class_map
    name2id = {v: int(k) for k, v in class_map[args.class_map].items()}
    organs = ABDOMEN if args.organs == "abdomen" else (
        list(name2id) if args.organs == "all" else
        TASKS[args.organs]["labels"] if args.organs in TASKS else args.organs.split(","))

    pred_img = nib.load(args.pred)
    pred = np.asanyarray(pred_img.dataobj).astype(np.int16)
    spacing = tuple(float(s) for s in pred_img.header.get_zooms()[:3])
    seg_dir = Path(args.gt_dir) / "segmentations"

    new = not Path(args.out).exists()
    with open(args.out, "a", newline="") as f:
        cols = ["subject", "tag", "organ", "gt_voxels", "pred_voxels", "dice", "nsd2mm"] + (["nsd3mm"] if args.nsd3 else [])
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        for organ in organs:
            gt_path = seg_dir / f"{organ}.nii.gz"
            if not gt_path.exists():
                continue
            gt = np.asanyarray(nib.load(gt_path).dataobj) > 0
            assert gt.shape == pred.shape, f"shape mismatch {gt.shape} vs {pred.shape}"
            pr = pred == name2id[organ]
            g, p = gt.sum(), pr.sum()
            if g == 0 and p == 0:
                dice = np.nan
            else:
                dice = 2 * (gt & pr).sum() / (g + p)
            gc, pc = bbox_crop(gt, pr)
            row = {"subject": Path(args.gt_dir).name, "tag": args.tag, "organ": organ,
                   "gt_voxels": int(g), "pred_voxels": int(p),
                   "dice": round(float(dice), 4), "nsd2mm": round(float(nsd(gc, pc, spacing)), 4)}
            if args.nsd3:
                row["nsd3mm"] = round(float(nsd(gc, pc, spacing, tol_mm=3.0)), 4)
            w.writerow(row)


if __name__ == "__main__":
    main()

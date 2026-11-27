"""Benchmark TotalSegmentator v3.1 (branch totalseg_v3, nnU-Net streaming inference) on the test scans.

Needs TotalSegmentator from its totalseg_v3 branch (v3.1). Two ways of running, both end to end (file in, file out):
  per-scan : one totalsegmentator() call per scan, like the command line (models loaded every call)
  folder   : one call on a folder of scans (models loaded once), time divided by the number of scans

Predictions are relabelled to the TotalSegmentator 2.18 'total_v3' ids used by eval_seg.py
(--id-map: JSON name -> id exported from the 2.18 environment).

  python ts31_bench.py --images @ct_list.txt --out-dir results/ts31 --id-map ids.json
         [--roi-subset abdomen14|spine26|a,b,c] [--tag ts31] [--folder] [--device gpu|cpu] [--limit N]
"""
import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path


import nibabel as nib
import numpy as np
import torch
from totalsegmentator.map_to_binary import class_map
from totalsegmentator.python_api import totalsegmentator


def stem_of(p):
    p = Path(p)
    return p.parent.name if p.name == "ct.nii.gz" else p.name.split(".")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--id-map", required=True)
    ap.add_argument("--tag", default="ts31")
    ap.add_argument("--roi-subset", default="")
    ap.add_argument("--folder", action="store_true", help="also time folder mode (models loaded once)")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    imgs = [l.strip() for x in a.images for l in (Path(x[1:]).read_text().split() if x.startswith("@") else [x])
            if l.strip()]
    if a.limit:
        imgs = imgs[:a.limit]
    roi = None
    if a.roi_subset:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from labels import TASKS
        roi = TASKS[a.roi_subset]["labels"] if a.roi_subset in TASKS else a.roi_subset.split(",")
        roi = [r for r in roi if r in class_map["total"].values()]
    out = Path(a.out_dir)
    pred_dir = out / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    target = json.loads(Path(a.id_map).read_text())  # name -> id (2.18 total_v3)
    lut = np.zeros(256, np.uint8)
    for i, name in class_map["total"].items():
        lut[int(i)] = target.get(name, 0)  # a structure unknown to 2.18 is dropped (not evaluated)
    cuda = a.device == "gpu" and torch.cuda.is_available()

    def run(inp, outp):
        return totalsegmentator(inp, outp, ml=True, task="total", roi_subset=roi, quiet=True, device=a.device)

    tpath = out / "timing.csv"
    new = not tpath.exists()
    with open(tpath, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "tag", "mode", "total", "peak_vram_gb"])
        if new:
            w.writeheader()
        for k, img in enumerate([imgs[0]] + imgs):  # first = warm-up, not recorded
            s = stem_of(img)
            tmp = pred_dir / f"{s}_{a.tag}_raw.nii.gz"
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            run(img, tmp)
            if cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if k == 0:
                print(f"warm-up {s}: {dt:.1f}s", flush=True)
                tmp.unlink(missing_ok=True)
                continue
            o = nib.load(tmp)
            lab = lut[np.asanyarray(o.dataobj).astype(np.uint8)]
            nib.save(nib.Nifti1Image(lab, o.affine, o.header), pred_dir / f"{s}_{a.tag}.nii.gz")
            tmp.unlink()
            row = {"image": s, "tag": a.tag, "mode": "per-scan", "total": round(dt, 2),
                   "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else 0.0}
            w.writerow(row); f.flush()
            print(row, flush=True)

        if a.folder:
            fin, fout = out / "folder_in", out / "folder_out"
            shutil.rmtree(fin, ignore_errors=True); shutil.rmtree(fout, ignore_errors=True)
            fin.mkdir()
            for img in imgs:  # hard links: no copies on the same drive
                os.link(img, fin / f"{stem_of(img)}.nii.gz")
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            run(fin, fout)
            if cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            row = {"image": f"folder of {len(imgs)}", "tag": a.tag + "_folder", "mode": "folder", "total": round(dt / len(imgs), 2),
                   "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else 0.0}
            w.writerow(row); f.flush()
            print(row, flush=True)
            shutil.rmtree(fin, ignore_errors=True); shutil.rmtree(fout, ignore_errors=True)
    print("TS31 BENCH DONE", flush=True)


if __name__ == "__main__":
    main()

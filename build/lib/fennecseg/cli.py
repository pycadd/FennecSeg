"""fennecseg command line.

  fennecseg predict scan.nii.gz -o seg.nii.gz --model models/fennecseg-abdomen14-v0.1.0
  fennecseg predict dicom_folder/ other.nii.gz -o out_dir/ [--split] [--lcc] [--label-ids totalsegmentator]
  fennecseg info --model models/fennecseg-abdomen14-v0.1.0

--model can also come from the FENNECSEG_MODEL environment variable.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from . import __version__
from .inference import predict
from .io import read_image, stem
from .model import load


def _device(name):
    if name == "auto":  # Apple MPS is not used: it lacks some 3D ops this pipeline needs
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def cmd_predict(a):
    dev = _device(a.device)
    torch.backends.cudnn.benchmark = dev.type == "cuda"
    t = time.perf_counter()
    model = load(a.model, device=dev)
    print(f"{model.manifest['name']} {model.manifest['version']} on {dev} (loaded in {time.perf_counter()-t:.1f}s)")
    label_map = None
    if a.label_ids == "totalsegmentator":
        label_map = model.manifest["label_ids_totalsegmentator"]
        label_map = {i: label_map[n] for i, n in model.labels.items()}
    out = Path(a.output)
    single = len(a.inputs) == 1 and out.name.endswith((".nii", ".nii.gz")) and not a.split
    if not single:
        out.mkdir(parents=True, exist_ok=True)
    writer = None
    if a.timing:
        new = not Path(a.timing).exists()
        tf = open(a.timing, "a", newline="")
    names = [stem(p) for p in a.inputs]
    if len(set(names)) < len(names):  # e.g. many <case>/ct.nii.gz: prefix the parent folder
        names = [f"{Path(p).resolve().parent.name}_{n}" for p, n in zip(a.inputs, names)]
    name_of = dict(zip(a.inputs, names))
    jobs = ([a.inputs[0]] if a.warmup and dev.type == "cuda" else []) + list(a.inputs)
    for k, path in enumerate(jobs):
        warm = a.warmup and dev.type == "cuda" and k == 0
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        T = {}
        t0 = time.perf_counter()
        img = read_image(path)
        T["load"] = time.perf_counter() - t0
        seg = predict(model, img, patch=a.patch, step=a.step, lcc=a.lcc, label_map=label_map, timings=T,
                      min_mm3=a.min_mm3)
        st = time.perf_counter()
        name = name_of[path]
        if single:
            nib.save(seg, out)
        elif a.split:
            d = out / name
            d.mkdir(exist_ok=True)
            arr = np.asanyarray(seg.dataobj)
            ids = label_map or {i: i for i in model.labels}
            for i, organ in model.labels.items():
                nib.save(nib.Nifti1Image((arr == ids[i]).astype(np.uint8), seg.affine), d / f"{organ}.nii.gz")
        else:
            nib.save(seg, out / f"{name}_seg.nii.gz")
        T["save"] = time.perf_counter() - st
        total = time.perf_counter() - t0
        if warm:
            print(f"warm-up {name}: {total:.2f}s")
            continue
        vram = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0
        print(f"{name}: {total:.2f}s (network {T['network']:.2f}s, resampled={T['resampled']}, "
              f"peak VRAM {vram:.2f} GB)")
        if a.timing:
            row = {"image": name, "tag": a.tag, "shape": "x".join(map(str, img.shape[:3])),
                   "spacing": "x".join(f"{z:.3g}" for z in img.header.get_zooms()[:3]), "total": total, **T,
                   "peak_vram_gb": vram}
            row = {k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in row.items()}
            if writer is None:
                writer = csv.DictWriter(tf, fieldnames=list(row))
                if new:
                    writer.writeheader()
            writer.writerow(row)
            tf.flush()


def cmd_info(a):
    print(load(a.model, device="cpu").describe())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fennecseg", description=f"fennecseg {__version__}: fast 3D CT segmentation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="segment CT scans")
    p.add_argument("inputs", nargs="+", help="NIfTI files, DICOM series folders, or NRRD/MHA files")
    p.add_argument("-o", "--output", required=True, help="output .nii.gz (one input) or folder")
    p.add_argument("--model", default=os.environ.get("FENNECSEG_MODEL"), required="FENNECSEG_MODEL" not in os.environ)
    p.add_argument("--device", default="auto", help="auto, cuda, cpu")
    p.add_argument("--patch", type=int, default=None, help="window size (default from the model)")
    p.add_argument("--step", type=float, default=None, help="window step as a fraction of the patch")
    p.add_argument("--lcc", action=argparse.BooleanOptionalAction, default=None,
                   help="keep the largest piece per organ (default from the model)")
    p.add_argument("--min-mm3", type=float, default=None,
                   help="drop pieces smaller than this volume in mm^3 (default from the model, 0 = off)")
    p.add_argument("--label-ids", choices=["model", "totalsegmentator"], default="model")
    p.add_argument("--split", action="store_true", help="one file per organ, in <output>/<scan>/")
    p.add_argument("--timing", default=None, help="append per-scan timings to this CSV")
    p.add_argument("--tag", default="fennecseg", help="tag column in the timing CSV")
    p.add_argument("--warmup", action="store_true", help="run the first scan twice, time the second (GPU)")
    p.set_defaults(fn=cmd_predict)
    i = sub.add_parser("info", help="describe a model package")
    i.add_argument("--model", default=os.environ.get("FENNECSEG_MODEL"), required="FENNECSEG_MODEL" not in os.environ)
    i.set_defaults(fn=cmd_info)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

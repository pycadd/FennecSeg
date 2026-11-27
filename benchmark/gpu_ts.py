"""TotalSegmentator 'total' (default, 5 part models @1.5 mm) run in one process, kept on the GPU.

Same weights as TotalSegmentator, so accuracy should match; only the plumbing changes:
  - all 5 networks loaded once and kept in VRAM (not reloaded per scan)
  - no temp NIfTI files, no nnU-Net worker processes
  - CT normalisation, sliding window, argmax, label merging and resampling on the GPU
  - logits never leave the GPU (nnU-Net copies all class logits to the CPU and post-processes there)

--breakdown additionally runs nnU-Net's own in-process CPU preprocessing and logits->segmentation
export for each part, timed, to explain where TotalSegmentator's time goes (results not used).

Usage:
  python gpu_ts.py --images a.nii.gz ... --out timing.csv --pred-dir preds [--breakdown] [--tag gpu]
"""
import argparse
import csv
import os
import time
from pathlib import Path


import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from totalsegmentator.config import setup_nnunet

setup_nnunet()
import totalsegmentator.nnunet  # noqa: F401  (registers TotalSegmentator's custom trainers / runtime patches)
from totalsegmentator.map_to_binary import class_map, class_map_5_parts, class_map_5_parts_total_v3
from totalsegmentator.map_to_binary import map_taskid_to_partname_ct, map_taskid_to_partname_ct_v3
from totalsegmentator.alignment import as_closest_canonical, undo_canonical
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

TASKS = {  # task -> (part task ids, part class maps, task id -> part name)
    "total": ([291, 292, 293, 294, 295], class_map_5_parts, map_taskid_to_partname_ct),
    "total_v3": ([831, 832, 833, 834, 835], class_map_5_parts_total_v3, map_taskid_to_partname_ct_v3),
}
TRAINER = "nnUNetTrainerNoMirroring"
SPACING = 1.5
STEP = 0.8  # TotalSegmentator uses tile step 0.8 for 'total'
DEV = torch.device("cuda")


def sync():
    torch.cuda.synchronize()


class Part:
    def __init__(self, tid, task, plans="nnUNetPlans"):
        self.tid = tid
        p = nnUNetPredictor(tile_step_size=STEP, use_gaussian=True, use_mirroring=False,
                            perform_everything_on_device=True, device=DEV, verbose=False,
                            verbose_preprocessing=False, allow_tqdm=False)
        p.initialize_from_trained_model_folder(get_output_folder(tid, TRAINER, plans, "3d_fullres"),
                                               use_folds=[0], checkpoint_name="checkpoint_final.pth")
        assert len(p.list_of_parameters) == 1
        p.network.load_state_dict(p.list_of_parameters[0])
        p.network.eval()
        self.p = p
        props = p.plans_manager.foreground_intensity_properties_per_channel["0"]
        self.lo, self.hi = props["percentile_00_5"], props["percentile_99_5"]
        self.mean, self.std = props["mean"], max(props["std"], 1e-8)
        assert p.configuration_manager.normalization_schemes == ["CTNormalization"], p.configuration_manager.normalization_schemes
        assert np.allclose(p.configuration_manager.spacing, [SPACING] * 3), p.configuration_manager.spacing
        # part label -> global 'total' label
        _, part_maps, tid2part = TASKS[task]
        inv = {v: k for k, v in class_map[task].items()}
        part_map = part_maps[tid2part[tid]]
        lut = torch.zeros(max(part_map) + 1, dtype=torch.uint8)
        for j, name in part_map.items():
            lut[j] = inv[name]
        self.lut = lut.to(DEV)

    @torch.inference_mode()
    def predict(self, x):
        """x: (1, z, y, x) float32 on GPU in nnU-Net axis order -> global labels uint8 (z, y, x) on GPU."""
        xn = (x.clamp(self.lo, self.hi) - self.mean) / self.std
        logits = self.p.predict_sliding_window_return_logits(xn)
        return self.lut[logits.argmax(0)]


def zoom_shape(shape, zooms, target):
    return [int(round(s * z / target)) for s, z in zip(shape, zooms)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--tag", default="gpu")
    ap.add_argument("--breakdown", action="store_true")
    ap.add_argument("--task", default="total", choices=list(TASKS))
    ap.add_argument("--parts", default="", help="comma-separated part task ids to run (default: all 5)")
    ap.add_argument("--plans", default="nnUNetPlans",
                    help="nnUNetResEncUNetLPlans_8 = TotalSegmentator v3 model_size='small'")
    args = ap.parse_args()
    args.images = [l.strip() for a in args.images for l in (Path(a[1:]).read_text().split() if a.startswith("@") else [a]) if l.strip()]  # @list.txt = one path per line
    Path(args.pred_dir).mkdir(parents=True, exist_ok=True)

    sync(); st = time.perf_counter()
    keep = {int(t) for t in args.parts.split(",") if t}
    parts = [Part(t, args.task, args.plans) for t in TASKS[args.task][0] if not keep or t in keep]
    print("parts:", [p.tid for p in parts], flush=True)
    sync(); load_s = time.perf_counter() - st
    print(f"loaded {len(parts)} models ({args.plans}) in {load_s:.1f}s, VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    fields = ["image", "tag", "shape", "total", "load", "to_gpu_resample_in", "network_all_parts",
              "merge_resample_out", "save", "peak_vram_gb",
              "nnunet_cpu_preprocess", "nnunet_logits_to_cpu", "nnunet_cpu_export"]
    new = not Path(args.out).exists()
    f = open(args.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()

    images = [args.images[0]] + list(args.images)  # first image twice: warm-up (cuDNN autotune)
    for i, img_path in enumerate(images):
        stem = Path(img_path).parent.name if Path(img_path).name == "ct.nii.gz" else Path(img_path).name.split(".")[0]
        torch.cuda.reset_peak_memory_stats()
        T = {}
        sync(); t0 = time.perf_counter()

        # 1. load + canonical (RAS), as TotalSegmentator does
        img = nib.load(img_path)
        can = as_closest_canonical(img)
        data = np.asanyarray(can.dataobj, dtype=np.float32)
        zooms = can.header.get_zooms()[:3]
        T["load"] = time.perf_counter() - t0

        # 2. to GPU, resample to 1.5 mm (trilinear) if needed, nnU-Net axis order (z, y, x)
        st = time.perf_counter()
        x = torch.from_numpy(data).to(DEV)
        tgt = zoom_shape(x.shape, zooms, SPACING)
        if list(x.shape) != tgt:
            x = F.interpolate(x[None, None], size=tgt, mode="trilinear", align_corners=False)[0, 0]
        x = x.permute(2, 1, 0).contiguous()[None]
        sync(); T["to_gpu_resample_in"] = time.perf_counter() - st

        # 3. five networks; later parts overwrite earlier ones where they predict foreground (as TotalSegmentator)
        st = time.perf_counter()
        seg = torch.zeros(x.shape[1:], dtype=torch.uint8, device=DEV)
        for part in parts:
            lab = part.predict(x)
            seg = torch.where(lab != 0, lab, seg)
        sync(); T["network_all_parts"] = time.perf_counter() - st

        # 4. back to RAS axis order, nearest-neighbour resample to the original grid, undo canonical
        st = time.perf_counter()
        seg = seg.permute(2, 1, 0)
        if list(seg.shape) != list(can.shape[:3]):
            seg = F.interpolate(seg[None, None].float(), size=list(can.shape[:3]), mode="nearest")[0, 0].to(torch.uint8)
        seg_np = seg.cpu().numpy()
        out_img = undo_canonical(nib.Nifti1Image(seg_np, can.affine), img)
        T["merge_resample_out"] = time.perf_counter() - st

        # 5. save
        st = time.perf_counter()
        nib.save(out_img, Path(args.pred_dir) / f"{stem}_{args.tag}.nii.gz")
        T["save"] = time.perf_counter() - st
        sync(); total = time.perf_counter() - t0

        row = {"image": stem, "tag": args.tag, "shape": "x".join(map(str, img.shape[:3])), "total": total,
               "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9, **T}

        # optional: what nnU-Net's own CPU steps cost on the same input (in-process, no worker spawn)
        if args.breakdown and i > 0:
            pre = l2c = exp = 0.0
            arr = data if list(data.shape) == zoom_shape(data.shape, zooms, SPACING) else None
            if arr is not None:
                raw = arr.transpose(2, 1, 0)[None]
                props = {"spacing": [float(z) for z in zooms[::-1]]}
                for part in parts:
                    p = part.p
                    pp = p.configuration_manager.preprocessor_class(verbose=False)
                    st = time.perf_counter()
                    d, _, prop = pp.run_case_npy(raw.copy(), None, dict(props), p.plans_manager,
                                                 p.configuration_manager, p.dataset_json)
                    pre += time.perf_counter() - st
                    with torch.inference_mode():
                        lg = p.predict_sliding_window_return_logits(torch.from_numpy(d).to(DEV))
                        sync(); st = time.perf_counter()
                        lg = lg.to("cpu")
                        l2c += time.perf_counter() - st
                    st = time.perf_counter()
                    convert_predicted_logits_to_segmentation_with_correct_shape(
                        lg, p.plans_manager, p.configuration_manager, p.label_manager, prop, False)
                    exp += time.perf_counter() - st
                    del lg
            row.update({"nnunet_cpu_preprocess": pre, "nnunet_logits_to_cpu": l2c, "nnunet_cpu_export": exp})

        row = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}
        if i == 0:
            print(f"warm-up {stem}: {total:.1f}s", flush=True)
            continue
        w.writerow(row); f.flush()
        print(row, flush=True)
    f.close()


if __name__ == "__main__":
    main()

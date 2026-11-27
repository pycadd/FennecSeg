"""Whole-scan inference, all on one device: resample to the model spacing, crop to the body, sliding window,
map back to the input grid, optional largest-component cleanup, return labels in the input's orientation.

Input at the model spacing (within 1%) skips both resampling steps, which is exactly the benchmarked path.
"""
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from .model import Model


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def gaussian_map(patch, device, dtype, sigma_scale=0.125):
    g = [torch.exp(-0.5 * ((torch.arange(p, device=device) - (p - 1) / 2) / (p * sigma_scale)) ** 2) for p in patch]
    w = g[0][:, None, None] * g[1][None, :, None] * g[2][None, None, :]
    w = w / w.max()
    return w.clamp_min(w[w > 0].min()).to(dtype)


def _starts(size, patch, step):
    if size <= patch:
        return [0]
    n = int(np.ceil((size - patch) / (patch * step))) + 1
    return [int(round(i * (size - patch) / (n - 1))) for i in range(n)]


@torch.inference_mode()
def sliding_window(net, x, patch, step, num_classes, return_probs=False):
    """x: (1, 1, X, Y, Z) normalised. Returns argmax labels (X, Y, Z) uint8, or class probabilities (C, X, Y, Z)."""
    dtype = torch.float16 if x.is_cuda else torch.float32
    shape = x.shape[2:]
    pad = [max(0, p - s) for p, s in zip(patch, shape)]
    if any(pad):
        x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]), value=float(x.min()))
    S = x.shape[2:]
    acc = torch.zeros((num_classes, *S), dtype=dtype, device=x.device)
    wsum = torch.zeros(S, dtype=dtype, device=x.device) if return_probs else None
    w = gaussian_map(patch, x.device, dtype)
    for a in _starts(S[0], patch[0], step):
        for b in _starts(S[1], patch[1], step):
            for c in _starts(S[2], patch[2], step):
                sl = (slice(a, a + patch[0]), slice(b, b + patch[1]), slice(c, c + patch[2]))
                with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
                    out = net(x[(slice(None), slice(None)) + sl])[0]
                acc[(slice(None),) + sl] += torch.softmax(out.float(), 0).to(dtype) * w
                if return_probs:
                    wsum[sl] += w
    crop = (slice(0, shape[0]), slice(0, shape[1]), slice(0, shape[2]))
    if return_probs:
        return acc[(slice(None),) + crop] / wsum[crop]
    return acc.argmax(0).to(torch.uint8)[crop]


def body_box(ct, thr=-500.0, margin=8):
    """Bounding box of voxels above thr HU (the body)."""
    m = ct > thr
    box = []
    for d in range(3):
        other = tuple(i for i in range(3) if i != d)
        nz = torch.nonzero(m.any(dim=other)).flatten()
        if len(nz) == 0:
            return None
        lo, hi = int(nz[0]) - margin, int(nz[-1]) + margin + 1
        box.append(slice(max(lo, 0), min(hi, ct.shape[d])))
    return tuple(box)


def _axis(a, b, L, N, dev):
    """Native voxels j whose model-grid position falls in crop [a, b) of a model grid of size L, and their
    grid_sample coordinates relative to the crop. Same convention as F.interpolate(align_corners=False)."""
    j0, j1 = max(0, int(np.floor(a * N / L))), min(N, int(np.ceil(b * N / L)))
    j = torch.arange(j0, j1, device=dev, dtype=torch.float32)
    s = (j + 0.5) * L / N - 0.5 - a
    return slice(j0, j1), (2 * s + 1) / (b - a) - 1


@torch.inference_mode()
def upsample_argmax(probs, box, L, N, slab_voxels=8_000_000):
    """probs (C, crop) on the model grid -> uint8 labels on the native grid N, trilinear on probabilities."""
    dev = probs.device
    gdtype = torch.float16 if probs.is_cuda else torch.float32
    out = torch.zeros(N, dtype=torch.uint8, device=dev)
    (sx, ux), (sy, uy), (sz, uz) = [_axis(b.start, b.stop, l, n, dev) for b, l, n in zip(box, L, N)]
    src = probs[None].to(gdtype)
    nz = max(1, slab_voxels // max(1, len(ux) * len(uy)))
    for k in range(0, len(uz), nz):
        uzc = uz[k:k + nz]
        gx, gy, gz = torch.meshgrid(ux, uy, uzc, indexing="ij")
        grid = torch.stack([gz, gy, gx], -1)[None].to(gdtype)  # grid_sample wants (W, H, D) = (z, y, x)
        p = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
        out[sx, sy, sz.start + k: sz.start + k + len(uzc)] = p.argmax(0).to(torch.uint8)
    return out


def keep_largest(lab: np.ndarray, ids) -> np.ndarray:
    """Per organ id, keep only its largest connected component (in place)."""
    from scipy import ndimage
    boxes = ndimage.find_objects(lab)
    for i in ids:
        if i > len(boxes) or boxes[i - 1] is None:
            continue
        sl = boxes[i - 1]
        m = lab[sl] == i
        cc, n = ndimage.label(m)
        if n <= 1:
            continue
        keep = np.argmax(np.bincount(cc.ravel())[1:]) + 1
        lab[sl][m & (cc != keep)] = 0
    return lab


def remove_small(lab: np.ndarray, ids, min_voxels: int) -> np.ndarray:
    """Per label id, drop connected components (26-connected) smaller than min_voxels (in place).
    Removes small isolated false detections, mostly of structures that are absent from the scan."""
    from scipy import ndimage
    if min_voxels <= 1:
        return lab
    boxes = ndimage.find_objects(lab)
    st = np.ones((3, 3, 3), bool)
    for i in ids:
        if i > len(boxes) or boxes[i - 1] is None:
            continue
        sl = boxes[i - 1]
        m = lab[sl] == i
        cc, n = ndimage.label(m, structure=st)
        sizes = np.bincount(cc.ravel())
        small = np.flatnonzero(sizes < min_voxels)
        small = small[small > 0]
        if len(small):
            lab[sl][m & np.isin(cc, small)] = 0
    return lab


def predict(model: Model, img: nib.Nifti1Image, patch=None, step=None, lcc=None, label_map=None, timings=None,
            min_mm3=None):
    """Segment one CT. Returns a uint8 label image on the input grid and orientation.
    label_map: optional {model id: output id} (e.g. TotalSegmentator ids). lcc: None = model default.
    min_mm3: drop pieces of any structure smaller than this volume in mm^3; None = model default, 0 = off."""
    T = timings if timings is not None else {}
    man = model.manifest
    dev = next(model.net.parameters()).device
    inf = man["inference"]
    patch = patch or inf["patch"]
    step = step or inf["step"]
    lcc = inf.get("largest_component", False) if lcc is None else lcc
    min_mm3 = man.get("postprocess", {}).get("min_component_mm3", 0) if min_mm3 is None else min_mm3
    ncls = man["architecture"]["num_classes"]
    inten = man["input"]["intensity"]

    st = time.perf_counter()
    can = nib.as_closest_canonical(img)
    sp = np.array(can.header.get_zooms()[:3], dtype=float)
    data = np.asanyarray(can.dataobj, dtype=np.float32)
    T["read"] = time.perf_counter() - st

    st = time.perf_counter()
    ct = torch.from_numpy(data).to(dev)
    target = np.array(model.spacing, dtype=float)
    resample = bool(np.any(np.abs(sp - target) > 0.01 * target))
    N = tuple(ct.shape)
    if resample:
        L = tuple(max(1, int(round(n * s / t))) for n, s, t in zip(N, sp, target))
        x = F.interpolate(ct[None, None], size=L, mode="trilinear", align_corners=False)[0, 0]
    else:
        L, x = N, ct
    box = body_box(x, man["input"].get("body_threshold_hu", -500.0)) or tuple(slice(0, n) for n in L)
    xin = ((x[box].clamp(inten["p00_5"], inten["p99_5"]) - inten["mean"]) / inten["std"])[None, None]
    _sync(dev); T["prepare"] = time.perf_counter() - st
    T["resampled"] = resample
    T["crop_frac"] = xin.numel() / x.numel()

    st = time.perf_counter()
    res = sliding_window(model.net, xin, (patch,) * 3, step, ncls, return_probs=resample)
    _sync(dev); T["network"] = time.perf_counter() - st

    st = time.perf_counter()
    if resample:
        full = upsample_argmax(res, box, L, N)
    else:
        full = torch.zeros(N, dtype=torch.uint8, device=dev)
        full[box] = res
    seg = full.cpu().numpy()
    _sync(dev); T["restore"] = time.perf_counter() - st

    st = time.perf_counter()
    if lcc:
        names = set(man.get("postprocess", {}).get("largest_component", []))
        keep_largest(seg, [i for i, n in model.labels.items() if n in names])
    if min_mm3:
        voxel_mm3 = float(np.prod(np.array(can.header.get_zooms()[:3], dtype=float)))  # output grid = input grid
        remove_small(seg, list(model.labels), int(round(min_mm3 / voxel_mm3)))
    if label_map:
        lut = np.zeros(256, np.uint8)
        for k, v in label_map.items():
            lut[int(k)] = int(v)
        seg = lut[seg]
    out = nib.Nifti1Image(seg, can.affine)
    out.header.set_data_dtype(np.uint8)
    ornt = nib.orientations.ornt_transform(nib.io_orientation(can.affine), nib.io_orientation(img.affine))
    out = out.as_reoriented(ornt)
    T["post"] = time.perf_counter() - st
    return out

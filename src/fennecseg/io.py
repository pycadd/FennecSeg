"""Read CT from NIfTI, a DICOM series folder, or anything SimpleITK reads (NRRD, MHA, ...).
Everything is returned as a nibabel image so the rest of the pipeline has one code path."""
from pathlib import Path

import nibabel as nib
import numpy as np


def _sitk_to_nib(img) -> nib.Nifti1Image:
    import SimpleITK as sitk
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D image, got shape {arr.shape}")
    arr = np.ascontiguousarray(arr.transpose(2, 1, 0))  # -> (x, y, z), voxel order of the ITK index
    d = np.array(img.GetDirection()).reshape(3, 3)
    aff = np.eye(4)
    aff[:3, :3] = d @ np.diag(img.GetSpacing())
    aff[:3, 3] = img.GetOrigin()
    aff = np.diag([-1, -1, 1, 1]) @ aff  # ITK physical space is LPS, NIfTI is RAS
    return nib.Nifti1Image(arr, aff)


def read_dicom_series(folder: Path) -> nib.Nifti1Image:
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(str(folder))
    if not ids:
        raise ValueError(f"no DICOM series found in {folder}")
    files = {i: reader.GetGDCMSeriesFileNames(str(folder), i) for i in ids}
    sid = max(files, key=lambda i: len(files[i]))  # the series with most slices
    if len(ids) > 1:
        print(f"note: {len(ids)} series in {folder}, using the largest ({len(files[sid])} slices)")
    reader.SetFileNames(files[sid])
    return _sitk_to_nib(reader.Execute())  # SimpleITK applies rescale slope/intercept -> HU


def read_image(path) -> nib.Nifti1Image:
    path = Path(path)
    if path.is_dir():
        return read_dicom_series(path)
    if path.name.endswith((".nii", ".nii.gz")):
        return nib.load(path)
    import SimpleITK as sitk
    return _sitk_to_nib(sitk.ReadImage(str(path)))


def stem(path) -> str:
    path = Path(path)
    if path.is_dir():
        return path.name
    n = path.name
    for ext in (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd", ".dcm"):
        if n.endswith(ext):
            return n[: -len(ext)]
    return path.stem

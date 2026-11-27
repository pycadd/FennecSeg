# FennecSeg

Compact, fast 3D CT segmentation models, one small network per group of structures, and a library to run them
on any CT scan. No TotalSegmentator or nnU-Net installation needed.

| model | structures | Dice | NSD 2 mm | s/scan | GPU memory |
|---|---|---|---|---|---|
| `abdomen14` 0.3.0 | spleen, kidneys, gallbladder, liver, stomach, pancreas, adrenal glands, esophagus, duodenum, aorta, inferior vena cava, portal and splenic veins | 0.884 | 0.900 | 1.16 | 4.2 GB |
| `spine26` 0.1.0 | vertebrae C1–L6 and sacrum | 0.894 | 0.934 | 1.20 | 6.2 GB |
| `thorax7` 0.1.0 | five lung lobes, trachea, heart | 0.917 | 0.903 | 1.10 | 3.2 GB |

Benchmark: all 109 scans of the official test split of the TotalSegmentator v3 dataset (never seen in
training), NVIDIA RTX 4070, time from file in to file out. On the same scans and structures, TotalSegmentator
v3.1 (one network for 117 structures) scores Dice 0.881 (abdomen), 0.898 (spine) and 0.921 (thorax), in 4.3 s
per scan, or 1.76 s per scan on a folder, for all its structures. Details, ablations and limitations are in the
technical report.

## Install

    pip install "fennecseg[hub] @ git+https://github.com/pycadd/FennecSeg"

A CUDA GPU is recommended; the CPU works but is slower.

## Use

    fennecseg predict scan.nii.gz -o seg.nii.gz --model abdomen14            # latest release
    fennecseg predict scan.nii.gz -o seg.nii.gz --model abdomen14@0.3.0      # exact version
    fennecseg predict dicom_folder/ -o out/ --split --model thorax7          # one file per structure
    fennecseg predict scan.nii.gz -o seg.nii.gz --model spine26 --label-ids totalsegmentator
    fennecseg info --model abdomen14

```python
import fennecseg
model = fennecseg.load("abdomen14@0.3.0")
seg = fennecseg.predict(model, fennecseg.read_image("scan.nii.gz"))
```

- **Inputs:** NIfTI, a DICOM series folder, NRRD or MHA, at any spacing and orientation. The scan is resampled
  to the model's 1.5 mm grid on the GPU, and the labels are written on the input grid and in its orientation.
- **Models** are downloaded once from the Hugging Face Hub ([amine0110/fennecseg](https://huggingface.co/amine0110/fennecseg))
  and cached. `--model` also accepts a local model folder.
- **Label ids:** each model uses its own ids by default (`fennecseg info`); `--label-ids totalsegmentator`
  writes TotalSegmentator's ids, so existing TotalSegmentator workflows keep working.
- **Post-processing** comes with the model. abdomen14 0.3.0 removes pieces smaller than 337.5 mm³ (about
  0.3 ml), which drops small false detections but also real structures that small; `--min-mm3 0` keeps them.

## Model packages and versions

A model is a folder with `weights.safetensors` and `model.json` (architecture, label table, input spacing,
intensity normalisation, post-processing and its own benchmark). Each group of structures has its own version
line; a release is an immutable hub tag `<group>-v<x.y.z>`, and a new version is released only if it beats the
current one on the fixed benchmark.

## Reproducing the benchmark

`benchmark/` holds the evaluation code (Dice and normalised surface Dice per structure), the list of the 109
test scans, and the scripts used to run TotalSegmentator v3.1 and TotalSegmentator's networks for comparison.
See [benchmark/README.md](benchmark/README.md).

## Licence and attribution

Code and the released model weights: [Apache License 2.0](LICENSE), © 2026 PYCAD.

The models are trained on the TotalSegmentator dataset (CC BY 4.0): Wasserthal J. et al., *TotalSegmentator:
Robust Segmentation of 104 Anatomic Structures in CT Images*, Radiology: Artificial Intelligence, 2023,
doi:10.1148/ryai.230024. FennecSeg is not affiliated with the TotalSegmentator authors.

FennecSeg is a research tool. It is not a medical device and must not be used for diagnosis or treatment
decisions.

## Contact

PYCAD, [pycad.co](https://pycad.co). For custom models and commercial support: mohammed@pycad.co.

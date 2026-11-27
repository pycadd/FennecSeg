# Benchmark

Everything needed to reproduce the accuracy numbers of the FennecSeg technical report.

- `test_subjects.txt`: the 109 scans of the official test split of the TotalSegmentator v3 dataset.
- `eval_seg.py`: Dice and normalised surface Dice (2 mm, optionally 3 mm) per structure, for a multilabel
  prediction against the dataset's binary masks. Predictions use TotalSegmentator's `total_v3` label ids.
- `labels.py`: the structure groups of the released models.
- `total_v3_ids.json`: TotalSegmentator 2.18 `total_v3` label ids (name -> id).
- `ts31_bench.py`: runs TotalSegmentator v3.1 (branch `totalseg_v3`) per scan and on a folder, timed end to end,
  and relabels its output to `total_v3` ids.
- `gpu_ts.py`: runs TotalSegmentator's own networks in the same GPU pipeline as FennecSeg (separates the
  model's contribution from the pipeline's).

## Steps

1. Download the TotalSegmentator v3 dataset (CC BY 4.0) and list the CTs of the test scans, one path per line:

       sed 's|.*|/data/Totalsegmentator_dataset_v300/&/ct.nii.gz|' test_subjects.txt > ct_list.txt

2. FennecSeg, with TotalSegmentator label ids and a timing file:

       fennecseg predict $(cat ct_list.txt) -o preds/ --model abdomen14@0.3.0 \
           --label-ids totalsegmentator --timing timing.csv

3. Score each scan (results are appended to `metrics.csv`):

       for s in $(cat test_subjects.txt); do
         python eval_seg.py --pred preds/${s}_ct.nii.gz --gt-dir /data/Totalsegmentator_dataset_v300/$s \
             --out metrics.csv --organs abdomen14 --class-map total_v3 --tag fennecseg
       done

   Use `--organs spine26` or `--organs thorax7` for the other models. Output file names follow
   `fennecseg predict` (check `preds/`).

4. Report the mean over structures of the per-structure mean over scans. Pairs where a structure is in
   neither the reference nor the prediction are skipped; structures present in fewer than five test scans
   (vertebrae_L6) are left out of the means.

5. TotalSegmentator v3.1 for comparison:

       python ts31_bench.py --images @ct_list.txt --out-dir results/ts31 --id-map total_v3_ids.json --folder

Timings in the report were measured on an NVIDIA RTX 4070 (12 GB) with the GPU otherwise idle.

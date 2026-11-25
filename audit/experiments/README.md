# Experiment results and verification records

This directory collects the saved evidence for technical findings C2–C4 and the R2 comparison of original 3D input with video-reconstructed input. The original audit scores and the new R2 scores come from separate evaluations. Do not combine them into one table or count the retained pilot judgments twice.

| Evidence | Scope | Start here |
|---|---|---|
| C2–C4 technical checks | 307 matched A/B pairs, 100 pre-GMR provenance links, and 400 historical judgments | [Technical findings and fixes](../REBUTTAL.md), [portable evidence](../evidence/c2_c4.json), [source records](c2_c4/source_records.json.gz) |
| R2 full cohort | 100 paired cases at three stages; 600 judgments, including the original 120 pilot judgments | [Results and interpretation](r2_full100/RESULTS.md), [case table](r2_full100/case_results.csv), [stage table](r2_full100/stage_results.csv) |
| R2 pilot | Original 20 cases; retained to document the earlier analysis and its exact reuse | [Pilot report](r2_pilot/RESULTS.md), [pilot case table](r2_pilot/case_results.csv) |

## Main R2 result

A starts from original 3D motion. B starts from video reconstructed with GVHMR. Both paths use TMED, GMR, and the same frozen SuperMimic policy. Saved cases, instructions, editor seeds and settings, and policy settings are matched. Existing motions and rollouts were rendered and judged; no motion generation, training, or robot rollout was repeated for R2.

| Stage | A successes out of 100 | B successes out of 100 | B minus A, percentage points | Paired 95% interval |
|---|---:|---:|---:|---:|
| Edited human motion | 32 | 29 | -3 | -12 to +6 |
| G1 reference | 33 | 27 | -6 | -14 to +2 |
| Simulated execution | 27 | 24 | -3 | -11 to +5 |

Four of the 600 verdicts are uncertain; none are technical errors. Uncertain verdicts do not count as successes and remain in the fixed denominator. All three intervals include zero. The results address the requested intermediate comparisons but do not identify a definitive stage where video input reduces fidelity. A is a control, not a proposed improvement over B. After editing, the goal is to preserve the requested change through retargeting and execution; success need not increase at each stage.

The original historical counts remain **35/27/24/21** for pre-GMR/C/A/B. R2 uses new controlled renders and new judging calls, so its counts do not replace those historical scores. The full-cohort endpoint was fixed before the additional 80 cases were judged, after the pilot had been inspected. The analysis is exploratory. The cohort contains selected completed rollouts; it is not the full MotionFix test set. R1 human validation remains deferred.

## Verify locally

From the repository root, using standard-library Python:

```bash
python3 audit/evaluate.py
python3 -m audit.verify_evidence
python3 audit/experiments/verify.py
```

The first two commands verify the original audit table and the C2–C4 evidence. The third checks the package hashes, compressed source-record hashes, all released R2 verdicts and frame-record links, case tables, stage transitions, paired bootstrap intervals, and exact paired tests. It also verifies that all 120 pilot results were reused unchanged. It does not require ICE paths, credentials, licensed models, SciPy, or GPU libraries.

[verification.log](verification.log) records the portable experiment check. The C2–C4 directory also contains the [audit aggregation log](c2_c4/audit_evaluation.log), [evidence verification log](c2_c4/evidence_verification.log), and [regression-test log](c2_c4/tests.log), generated when this package was assembled. These are verification reruns, not new model experiments.

## What is archived

- `c2_c4/source_records.json.gz` contains original JSON text and its SHA-256 for the source reports, run specifications, per-case editor records, retargeting reports, execution results and metadata, render preparation records, judging tasks, request metadata, and results used for C2–C4. `source_path` is relative to the original ICE project. The filename refers to the contents; some original content-addressed files have no extension.
- Each R2 directory contains the frozen protocol, original experiment and validation scripts, report, case and stage tables, and compressed results and input records. Decompress `*.json.gz` with Python's `gzip` module or `gzip -dk` before inspecting them as JSON.
- `response_records.json.gz` preserves the exact returned verdict text, model, response status, usage, and original response hash. It is an explicit field-selected export, not a full API envelope.
- R2 `request_records.json.gz` preserves the prompt, instruction, frame timestamps, image-detail setting, request settings, and saved request metadata. Image bytes are replaced by image hashes, byte counts, and MIME types. The original full request bytes were verified against their recorded hashes on ICE before this export. The public file cannot by itself reconstruct the original request bytes.
- R2 `video_records.json.gz` preserves the original rendering metadata text and hashes. The input records retain the source-video metadata. Videos themselves are not included.
- The R2 `logs/` directories contain the original Slurm stdout logs. [slurm_accounting.log](slurm_accounting.log) records the job states. The full-cohort H100 request and its dependent QA job were cancelled while pending; the completed render used a V100. The pilot used an H100. Software and rendering computations were unchanged, as recorded in [implementation_conformance.json](r2_full100/implementation_conformance.json).
- [export_summary.json](export_summary.json) records the export counts. [manifest.json](manifest.json) gives file sizes and SHA-256 values for the package, except itself and the generated verification log.

Credentials, authentication headers, API account metadata, encrypted reasoning, image/video bytes, motion arrays, licensed body models, and checkpoints are excluded. Original response hashes identify the complete envelopes retained on ICE. Historical C4 full image request payloads were never archived; their recorded hashes can be cross-checked but cannot be reconstructed from these logs. R2 full request payloads remain in the isolated ICE experiment directories.

## Original execution and export

The frozen R2 scripts and Slurm files retain their original ICE paths to preserve their recorded hashes. They require the original environments and assets; the portable verification command above is the entry point for reviewers. `r2_full100/build_extension.py` records how the extension was derived from the pilot. The published result reports contain the interpretation; original execution scripts are not modified to describe the response document.

The original project directory is `/storage/ice1/3/5/szhang957/Repos/VisualMimic-20261025`. The R2 directories are its siblings `visualmimic_r2_20261124` and `visualmimic_r2_full100_20261124`.

To regenerate the field-selected record archives from authorized access to those saved files:

```bash
python3 audit/experiments/export_records.py \
  --project /path/to/VisualMimic-20261025 \
  --pilot /path/to/visualmimic_r2_20261124 \
  --full /path/to/visualmimic_r2_full100_20261124 \
  --output /path/to/separate-export
```

The exporter reads original records and writes only to the designated output directory. It makes no model calls. C5's missing distance-matrix evidence and the other unaddressed research findings are not marked resolved by this release.

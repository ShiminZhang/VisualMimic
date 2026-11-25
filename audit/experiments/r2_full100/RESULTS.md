# VisualMimic R2 evaluation on the full 100-case cohort

This experiment extends the inspected 20-case pilot to every case in the original 100-case judging cohort. It retains all 120 pilot judgments and adds 480 judgments for the other 80 cases. The full-cohort endpoint was fixed before the additional scoring. R1 remains deferred. Existing project source files and audit/rebuttal documents were not edited.

## Design

A is the original 3D-input path. B is the saved video-reconstruction path. The same cases and instructions are compared after TMED, after GMR, and after SuperMimic execution. Saved run records verify matched editor settings, retargeting settings, policy checkpoints, and execution seeds. No motion reconstruction, editing, retargeting, training, or policy rollout was repeated.

The rendering computations, frame sampling, judge prompt, model identifier, request format, and response handling match the pilot. Human A/B outputs share a fixed camera within each case. All four robot reference/execution outputs share a fixed camera within the case. Original timing is retained. New requests use the same original source video for all six conditions and hide branch, stage, and historical verdict labels. GPT-6-Luna receives chronological frames at approximately 8 FPS and returns success, failure, or uncertain. Technical errors remain explicit.

The pilot renders used an H100. The additional renders used a V100 because the H100 request was pending. The same software environments and rendering computations were used, and all videos underwent metadata and visual checks. The original pilot videos and verdicts were reused exactly.

## Main experiment table

| Stage | A success | B success | B minus A (percentage points) | Paired 95% bootstrap interval | A-only / B-only successes |
|---|---:|---:|---:|---:|---:|
| After TMED: edited human motion | 32/100 (32%) | 29/100 (29%) | -3 | -12 to +6 | 12 / 9 |
| After GMR: G1 reference | 33/100 (33%) | 27/100 (27%) | -6 | -14 to +2 | 12 / 6 |
| After SuperMimic: recorded execution | 27/100 (27%) | 24/100 (24%) | -3 | -11 to +5 | 9 / 6 |

Intervals use 10,000 paired case-bootstrap resamples. Success rates retain all 100 cases in the denominator. Uncertain verdicts remain separate categories and do not count as successes. A-only/B-only counts refer to cases where only one branch received a success verdict.

| Condition | Success | Failure | Uncertain | Technical error |
|---|---:|---:|---:|---:|
| A_human | 32 | 68 | 0 | 0 |
| B_human | 29 | 70 | 1 | 0 |
| A_reference | 33 | 66 | 1 | 0 |
| B_reference | 27 | 72 | 1 | 0 |
| A_execution | 27 | 73 | 0 | 0 |
| B_execution | 24 | 75 | 1 | 0 |

## Paired statistical checks

| Stage | Exact paired-test p | Holm-adjusted p across three stages |
|---|---:|---:|
| After TMED: edited human motion | 0.663624 | 1 |
| After GMR: G1 reference | 0.237885 | 0.713654 |
| After SuperMimic: recorded execution | 0.607239 | 1 |

The exact paired binary tests and case-bootstrap intervals are the original exploratory analyses. Holm adjustment is also reported as a conservative check for the three stage comparisons. Neither analysis measures systematic VLM error or establishes causal attribution.

## Retained pilot and additional cases

| Stage | Pilot A / B success (n=20) | Additional A / B success (n=80) |
|---|---:|---:|
| After TMED: edited human motion | 9 / 9 | 23 / 20 |
| After GMR: G1 reference | 8 / 4 | 25 / 23 |
| After SuperMimic: recorded execution | 7 / 8 | 20 / 16 |

## Within-branch transitions

| Transition | A success to failure | A failure to success | B success to failure | B failure to success |
|---|---:|---:|---:|---:|
| human to reference | 10 | 12 | 10 | 7 |
| reference to execution | 13 | 6 | 12 | 9 |

## Interpretation for R2

The video-input branch has 3 percentage points lower judged success after editing, 6 points lower at the G1 reference, and 3 points lower after execution. The largest observed gap is at the G1 reference, but every paired 95% interval includes zero. None of the three exact paired comparisons is statistically significant, including before multiplicity adjustment. These results do not establish a consistent disadvantage from video input or identify a stage that causes such a disadvantage. They also do not establish equivalence between the branches.

The full cohort changes the pilot interpretation. The reference-stage gap narrows from 20 to 6 percentage points. The execution difference changes from a 5-point advantage for B to a 3-point disadvantage. The additional 80 cases therefore temper the apparent stage-specific pattern seen in the pilot.

The results support a narrower motivation for stage-wise and end-to-end evaluation. Ten cases in each branch change from judged success after editing to failure at the G1 reference. From reference to execution, 13 cases in A and 12 in B change from success to failure. Changes in the opposite direction also occur. Aggregate success is not monotonically decreasing: A changes from 32 to 33 to 27 successes, while B changes from 29 to 27 to 24. Thus, the evidence supports case-level changes in judged instruction following, rather than a claim that every pipeline stage necessarily reduces success.

This experiment directly addresses R2 by adding matched A/B comparisons at the two intermediate stages and retaining the execution comparison. It supports reporting what happens across the pipeline, but not a causal explanation of reconstruction or retargeting errors. The 27 A and 24 B execution successes provide automated evidence of instruction following on some cases in this selected cohort. They are not human-validated success rates. R1 remains unaddressed.

These are newly scored, controlled-view results. The historical audit scores came from a different evaluation presentation and should remain separately labeled; this experiment does not overwrite them.

## Scope and limitations

These results cover the entire existing cohort, but that cohort was previously selected and contains completed A/B rollouts. It is not the full MotionFix test set or an evaluation of completion failures. The expansion followed inspection of the pilot, so inference remains exploratory. Case-bootstrap uncertainty does not include VLM or rendering uncertainty. The model identifier is shared across requests; no stronger claim about an immutable backend model snapshot is made.

Human validation was not performed. Embodiment and rendering differ between human and robot stages, even though A/B settings are matched within each stage. A stage-wise difference does not establish a causal failure mechanism. These controlled-view results must remain separate from the historical audit table.

## Verification and files

The experiment contains 600 unique case-condition results. Saved request and response records, video identifiers, category counts, and paired counts were verified. All 120 pilot results match their originals, and 726 retained pilot files passed the byte-hash preservation check. An independent statistical implementation reproduced the exact paired-test p-values.

- [Frozen protocol](protocol.json)
- [Implementation comparison](implementation_conformance.json)
- [Stage results CSV](stage_results.csv)
- [Case results CSV](case_results.csv)
- [Full judgments and reasons](results.json.gz)
- [Statistical report](report.json)
- [Independent analysis](independent_analysis.json)
- [Record verification](verification.json)

The isolated ICE directory is `/storage/ice1/3/5/szhang957/Repos/visualmimic_r2_full100_20261124`. It holds the new videos, compressed request payloads, responses, metadata, and job logs. Retained pilot media and records remain in the original pilot directory and are linked from the extension.

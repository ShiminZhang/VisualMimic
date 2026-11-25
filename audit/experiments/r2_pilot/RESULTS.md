# VisualMimic R2 pilot results

The 20-case pilot found its largest A/B difference at the G1-reference stage. A and B had equal success totals after human-motion editing. B had fewer successes after retargeting, but this disadvantage did not persist after execution. The paired comparisons are inconclusive at this sample size. These are VLM judgments, not human-confirmed outcomes.

R1 was deferred at the user's request. Existing project source files and audit/rebuttal documents were not edited. All new experiment files are isolated in this experiment directory and its corresponding ICE directory.

## Artifact availability

The original ICE project remains available at `/storage/ice1/3/5/szhang957/Repos/VisualMimic-20261025`. All 100 cases in the previous judging cohort passed the file check. We read and verified SHA-256 hashes for 1,300 unique files: the source videos and both branches' execution results, retargeting reports, edited human motions, compiled G1 references, rollout trajectories, and execution videos. No missing files or hash mismatches were found.

## Frozen experiment design

- **Question:** At which observed stage do the 3D-input and video-input paths differ in instruction success?
- **Sample:** 20 distinct cases sampled uniformly from the original 100-case cohort, using seed 20261124. Selection was fixed before collecting new judgments. The parent cohort contains completed A/B rollouts and is not the full test set.
- **Conditions:** A uses the original 3D input. B uses the saved video-reconstruction path. Each case is evaluated after TMED, after GMR, and after SuperMimic execution, giving six conditions and 120 judgments.
- **Matched settings:** Cases, instructions, editor seed, editor checkpoint, retargeting settings, policy checkpoint, and execution seed are verified through the saved run records. The saved editor seed is 42 and execution seed is 0. No motion generation, reconstruction, retargeting, policy training, or policy rollout was repeated.
- **Rendering:** Both edited human motions share a fixed camera for each case. Both G1 references and both recorded executions share another fixed camera for that case. Camera bounds cover the compared motions. Rendering applies constant initial horizontal centering; human meshes also receive a constant floor shift. Temporal motion, orientation, relative displacement, and playback speed are preserved. Human clips use 30 FPS and robot clips use 25 FPS, with actual frame timestamps retained. The original source video is shared across all conditions for a case.
- **Judging:** All conditions were freshly evaluated using GPT-6-Luna, the same instruction-success prompt, medium reasoning effort, and a 2,048-token output limit. The judge receives source and candidate frames sampled at approximately 8 FPS, with timestamps. Requests omit case, branch, stage, and historical verdict labels. Request order is randomized with the frozen seed.
- **Outcomes:** Success requires the complete requested edit. Failure, uncertain, and technical-error outcomes remain separate. The denominator stays at 20 for every condition.
- **Analysis:** Report A/B success counts, the B-minus-A difference, paired case-bootstrap intervals using 10,000 resamples, and discordant success counts. Exact paired binary tests are exploratory and unadjusted for the three stage comparisons. Their uncertainty does not capture systematic VLM errors.

## Main experiment table

| Evaluation stage | A success | B success | B minus A | Paired 95% bootstrap interval | A-only / B-only successes |
|---|---:|---:|---:|---:|---:|
| After TMED: edited human motion | 9/20 (45%) | 9/20 (45%) | 0 percentage points | −20 to +20 points | 2 / 2 |
| After GMR: G1 reference | 8/20 (40%) | 4/20 (20%) | −20 percentage points | −40 to 0 points | 5 / 1 |
| After SuperMimic: recorded execution | 7/20 (35%) | 8/20 (40%) | +5 percentage points | −10 to +20 points | 1 / 2 |

All 120 judgments completed. There were zero uncertain outcomes and zero technical errors. All non-success outcomes in this run were explicit failure judgments. Exact paired-test p-values are 1.000, 0.21875, and 1.000, respectively. None establishes an A/B difference at the conventional 0.05 threshold.

## Within-branch transitions

| Transition | A success → failure | A failure → success | B success → failure | B failure → success |
|---|---:|---:|---:|---:|
| Edited human motion → G1 reference | 2 | 1 | 5 | 0 |
| G1 reference → recorded execution | 2 | 1 | 1 | 5 |

Equal editing-stage totals do not imply identical behavior: two cases succeed only in A and two only in B. At the reference stage, five cases succeed only in A and one only in B. This makes the reference stage a useful target for follow-up inspection, but it does not establish GMR as the cause of a systematic video-input disadvantage.

B's execution-stage increase comes from five failure-to-success judgments and one success-to-failure judgment. For example, in case 000101, “open legs less,” the judge describes a wide stance in B's reference and narrower or staggered legs in B's execution. This illustrates that execution can alter the feature used by the semantic judge. It does not establish improved tracking or overall robot quality. Other explanations, including judgment variability, remain possible.

## Comparison with historical judgments

These results must remain separate from the previous 100-case audit table. They use a smaller sample, controlled output cameras, and fresh judging calls.

| Shared condition on these same 20 cases | Historical successes | Fresh successes |
|---|---:|---:|
| A edited human motion | 11 | 9 |
| A G1 reference | 7 | 8 |
| A execution | 8 | 7 |
| B execution | 6 | 8 |

The execution ordering reverses between these two evaluations of the same saved motions. Across the four shared conditions, 68 of 80 labels agree and 12 change. Camera settings and judging calls both changed, so this comparison cannot isolate their individual effects. It reinforces the need to treat a one-case execution advantage cautiously.

## Interpretation and claim limits

This experiment directly addresses R2's request for intermediate A/B comparisons. The largest observed difference occurs at the G1-reference stage, while the execution-stage difference is small and has the opposite sign. The pilot does not support a monotonic loss across stages or a reliable claim that video input is worse after execution.

The evidence is limited to 20 cases from a previously selected completed-rollout cohort. All paired intervals include zero. Embodiment changes across stages, VLM errors, and dependence on rendering remain unresolved. Equal success totals do not establish equivalence, and stage-wise observations do not identify causal failure mechanisms. Human validation under R1 has not been performed.

## Verification and records

The video check covered all 120 candidates and 20 source videos, including hashes, timestamps, frame counts, first/middle/final-frame decoding, and matched camera metadata. Comparison sheets for all cases were visually inspected. All 120 saved request payloads and response records passed hash and label verification. An independent recount reproduced the group and paired counts, and an independent statistical implementation reproduced the exact paired-test p-values.

- [Frozen protocol](protocol.json)
- [Artifact availability check](artifact_availability.json)
- [Main results table as CSV](stage_results.csv)
- [Per-case labels as CSV](case_results.csv)
- [Full judgment results and reasons](results.json.gz)
- [Statistical report](report.json)
- [Verification report](verification.json)

The isolated ICE directory is `/storage/ice1/3/5/szhang957/Repos/visualmimic_r2_20261124`. It retains the rendered videos, full compressed request payloads, responses, metadata, and job logs. Completed jobs: rendering 7297240, video checks 7297283, judging 7297384, and verification 7297495.

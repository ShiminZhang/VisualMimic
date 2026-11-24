# Evidence for technical findings C2 to C4

The saved ICE records support the settings and provenance claims for the reported cohorts. The audit table values are unchanged. These checks read existing artifacts; they did not repeat training, reconstruction, editing, simulation, rendering, or VLM inference. The original audit packet and historical campaign files are retained unchanged.

| Finding | Check and result | Evidence and implementation |
|---|---|---|
| C2: matched settings | All 307 A/B pairs, including the 100 judged cases, share one recorded configuration. Editor seed is 42; execution seed is 0. Editor identity, generation options, GMR configuration and mapping, robot specifications, policy checkpoint, execution protocol, reference compiler, initial reference frame, and robot rendering settings match. | `c2` in [the evidence export](evidence/c2_c4.json); [ICE exporter](export_ice_evidence.py). Each edited output is linked to its saved editor item and run identity, then to the retargeting and execution records. |
| C3: pre-GMR provenance | All 100 pre-GMR clips match A's GMR input through the saved preparation record, canonical motion hash, video hash, timestamps, and render configuration. C references match A's compiled references. A/B candidate videos match the saved execution records. | `c3` in [the evidence export](evidence/c2_c4.json); [render provenance validator](../visualmimic/render_provenance.py). All three pre-GMR finalizers now require this validation before creating judging tasks. Preparation data remains outside the video-only reconstruction manifest. |
| C4: judgment accounting | All 400 saved API responses agree with their result records, combined report, and released CSV labels. Recorded source/candidate videos and frame indices match the saved judging tasks and 8 FPS sampling rule. | `c4` and `protocols` in [the evidence export](evidence/c2_c4.json); [strict cohort accounting](../visualmimic/judgment_accounting.py); [CSV validation and aggregation](evaluate.py). Exact returned verdict text, original protocol text, frame indices, and recorded request/response hashes are released. |

The common policy checkpoint SHA-256 is `387e32872046d7e7a618322dc9c832e1f021cf2aa389c3407143718075bed1e2`. These are comparisons of saved settings and linked outputs, not a claim that a new run would reproduce every output bit for bit. The exporter validates content-addressed JSON records and run identities. It does not reload model weights or execute the policy.

## Complete judgment counts

| Setting | Success | Failure | Uncertain | Technical error | Total |
|---|---:|---:|---:|---:|---:|
| Pre-GMR | 35 | 65 | 0 | 0 | 100 |
| C | 27 | 73 | 0 | 0 | 100 |
| A | 24 | 76 | 0 | 0 | 100 |
| B | 21 | 79 | 0 | 0 | 100 |

Zero uncertain/error counts describe this selected cohort. They do not imply that the full pipeline or earlier cohort selection had no failures. New summaries explicitly print all four categories, retain uncertain and technical-error outcomes in the denominator, and reject missing, duplicate, unexpected, or invalid records. The three retrieval columns remain the composite G1 proxy over 307 completed pairs.

## Verify without the experiment environment

From the repository root, using standard-library Python:

```bash
python3 audit/evaluate.py
python3 -m audit.verify_evidence
```

The second command checks the exported settings comparisons, cohort membership, motion/video/reference links, protocol-file hashes, exact verdict text, and agreement with the released CSV. It reports 307 matched pairs, 100 pre-GMR links, and 400 judgments. It does not read ICE paths or need model checkpoints.

The tests include swapped videos with unchanged sample keys and instructions, altered video bytes, duplicate keys, missing preparation records, changed editor seeds, invalid ranks, and incomplete/invalid judgment evidence:

```bash
python3 -m pytest -q tests
```

## Source records and regeneration

The original project root on ICE is `/storage/ice1/3/5/szhang957/Repos/VisualMimic-20261025`. The exporter starts from `campaigns/abc_g1_retrieval307_v1/parent_snapshot/inputs.json` for the 307 pairs and follows the recorded A/B result references to the run ledgers. The A editor run is `runs/a_test1013_gpu_v1`; the B editor run is `campaigns/b_test1013_v1/edited`. It reads the rendering records in `luna_pregmr10_v1`, `luna_extend30_v1`, and `luna_extend100_v1`, and the judge records in those campaigns plus `luna_abc10_v1` and the seven 100-case extension shards.

With access to those existing artifacts, run the revised code from a separate checkout:

```bash
python3 -m audit.export_ice_evidence /path/to/VisualMimic-20261025 > evidence.json
```

The exporter verifies the two source-report hashes recorded in [RESOURCES.md](RESOURCES.md) and fails on a mismatch. It streams a new evidence JSON to stdout and writes no campaign files. The portable export contains selected fields and original source-file hashes; it is not a replacement for the full ICE archive.

## Remaining limits

The export preserves the exact returned verdict text. Full API envelopes, account metadata, encrypted reasoning, videos, and licensed/model assets are not redistributed. Full HTTP image payloads were not archived, so request hashes were checked for agreement between saved records, not reconstructed from the original request bytes. These checks establish recorded provenance and accounting. They do not independently establish visual-judge accuracy, causal failure attribution, or full-test-set performance. Research findings 3 and 4 remain separate work.

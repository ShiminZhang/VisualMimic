# SuperMimic for G1 tracking

VisualMimic uses this independent SuperMimic implementation to track G1 motion references in mjlab / MuJoCo Warp. Training fits a world model to simulator transitions and updates the policy through imagined rollouts. The policy can also be loaded from a compatible checkpoint for [execution in VisualMimic](../README.md#run-the-learned-pipeline).

## Train and evaluate

Run the commands below from `supermimic/` on a Linux CUDA machine with Python 3.12 and uv. Provide retargeted G1 clips in LAFAN1 CSV format or converted 50 Hz NPZ files. For the asset-free CPU example, use the [repository quickstart](../README.md#run-the-cpu-smoke-test).

```bash
uv sync --frozen
uv run python scripts/data/csv_to_npz_batch.py --help
uv run sm-train --run-name lafan1 --data.npz-dir data/npz/lafan1 \
  --env.num-envs 4096 --policy-trainer.batch-size 4096 --schedule.max-hours 3
uv run python scripts/eval_benchmark.py --ckpt logs/lafan1/ckpt_latest.pt \
  --npz-dir data/npz/lafan1
```

The conversion code lives in `src/supermimic/data/`. Checkpoints and TensorBoard logs go in the run directory. When comparing runs, record whether anchor-relative observations were enabled. See the [experiment records](docs/EXPERIMENTS.md) and [reproduction plan](docs/REPRODUCTION_PLAN.md).

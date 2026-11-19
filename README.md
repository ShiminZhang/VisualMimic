# VisualMimic

How much does starting from video change the success of a text-guided motion edit? VisualMimic compares the **same instruction, editor, G1 retargeter and frozen SuperMimic policy** on two inputs. Branch A starts from a 3D human motion. Branch B estimates that motion from video with GVHMR. Branch C retains the G1 reference from A so you can inspect the edit before physics execution.

## Run the CPU smoke test

Use Python 3.12 on macOS or Linux. Run these commands in a shell:

```bash
git clone https://github.com/ShiminZhang/VisualMimic.git
cd VisualMimic
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-smoke.txt
PYTHONPATH=supermimic/src python -m examples.smoke --output smoke-run
python -m pytest -q tests
```

The example writes an 11-frame synthetic motion CSV, converts it with SuperMimic's CPU MuJoCo forward kinematics, and imports the result through VisualMimic's reference compiler. It checks the resampled joint angles and root trajectory. Expect **PASS: 21 frames at 50 Hz, 30 bodies**, a `smoke-run/reference.npz`, and a content-addressed copy in `smoke-run/artifacts/`. Choose a fresh output directory for each run. The test suite also exercises the 3D and video runners with fixture backends, frame conversion, ground placement and evaluation accounting.

The smoke uses a synthetic mechanism with G1 joint ordering and runs without checkpoints, CUDA, a dataset or API credentials. The learned editing and policy execution paths require the assets below.

## Run the learned pipeline

For reconstruction, editing and retargeting, follow the [single-sample commands](visualmimic/README.md). For G1 execution, provision a Linux CUDA machine with Python 3.12, uv, a compatible SuperMimic checkpoint and a compiled G1 reference NPZ (or an edited robot motion from the single-sample commands). From the repository root:

```bash
(cd supermimic && uv sync --frozen)
supermimic/.venv/bin/python -m visualmimic.execute_reference \
  --reference /path/to/reference.npz --checkpoint /path/to/checkpoint.pt \
  --implementation-commit "$(git rev-parse HEAD)" --device cuda:0 --output rollout
```

For a retargeted edit, replace `--reference /path/to/reference.npz` with `--robot-motion pair/edited_robot.npz`. Inspect `rollout/status.json` for the execution outcome and `rollout/result.json` for tracking metrics and artifact references. An early termination is a valid recorded outcome. Use the CPU smoke above for the independently verified path in this release.

[Interfaces](visualmimic/INTERFACES.md) · [Experiment design](VisualMimic.md) · [SuperMimic training](supermimic/README.md) · [Pinned upstream revisions and patches](dependencies.json)

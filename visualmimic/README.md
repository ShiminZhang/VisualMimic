# Run one motion edit

Run the commands from the repository root. Prepare the editor environment using the upstream MotionFix setup and `editor-requirements.txt`. Install GMR in that environment. Video reconstruction also needs a separate GVHMR GPU environment (see `gvhmr-requirements.txt`). Obtain the TMED checkpoint, CLIP configuration, SMPL-H body models and, for GVHMR, its model weights and neutral SMPL-X model through their respective providers.

Install the pinned source trees and apply their patches:

```bash
python - <<'PY'
import json, subprocess
from pathlib import Path
for name, dep in json.loads(Path("dependencies.json").read_text()).items():
    dest = Path("visualmimic/external") / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", dep["url"], dest], check=True)
    subprocess.run(["git", "-C", dest, "checkout", "--detach", dep["commit"]], check=True)
    if "patch" in dep:
        subprocess.run(["git", "-C", dest, "apply", Path(dep["patch"]).resolve()], check=True)
PY
```

For branch A, supply a single-sample JSON object containing `sample_id`, `instruction`, and `source`. The source is a VisualMimic motion artifact reference (`uri`, `sha256`, `media_type`) within the supplied artifact directory. Use an existing prepared manifest or prepare MotionFix inputs with `python -m visualmimic.prepare_motionfix --help`.

```bash
python -m visualmimic.prepare_robot_pair \
  --input source_manifest.json --source-artifacts source_artifacts \
  --motionfix-repository visualmimic/external/motionfix \
  --gmr-repository visualmimic/external/GMR \
  --checkpoint /path/to/tmed/last.ckpt --body-models /path/to/body_models \
  --clip-config /path/to/clip-vit-large-patch14 --output pair
```

The result includes `pair/edited_robot.npz` and its unedited counterpart. Use the [GPU execution command](../README.md#run-the-learned-pipeline) on the edited file.

For branch B, first run the following on a video manifest produced by `visualmimic.render_source`. The manifest carries the video artifact reference, timestamps, frame dimensions, sample identity and instruction.

```bash
python -m visualmimic.reconstruct_video \
  --input video/manifest.json --video-artifacts video/artifacts \
  --dataset-version "$DATASET_VERSION" --split test \
  --repository visualmimic/external/GVHMR --python /path/to/gvhmr/bin/python \
  --body-models /path/to/body_models --output reconstruction
```

Set `DATASET_VERSION` to the identifier used to prepare the sample. Pass `reconstruction/source_manifest.json` and `reconstruction/artifacts` to the same editor command and select a new output directory. Both branches use the same policy checkpoint during execution.

[Original design and implementation notes](https://github.com/ShiminZhang/VisualMimic/blob/b18f29a3aeda040d1f96a45d79b90df8239d7e0d/visualmimic/README.md)

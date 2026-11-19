"""Exercise the CPU motion-converter / tracking-reference boundary on synthetic input."""
import argparse
from pathlib import Path

import mujoco
import numpy as np
from supermimic.data.motion_convert import G1_JOINT_NAMES, convert_csv_to_npz
from visualmimic.reference_compiler import ReferenceCompiler, read_npz
from visualmimic.robot_contracts import RobotSpec
from visualmimic.storage import ArtifactStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    # Small synthetic mechanism with the G1 joint order. Each link has one hinge.
    links = "".join(
        f'<body name="link_{i}" pos="0 0 .02"><joint name="{name}"/>'
        '<geom type="sphere" size=".01"/></body>'
        for i, name in enumerate(G1_JOINT_NAMES)
    )
    xml = ('<mujoco><worldbody><body name="pelvis"><freejoint/>'
           '<geom type="sphere" size=".02"/>' + links + '</body></worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml)
    raw = np.zeros((11, 36))
    raw[:, 0] = np.linspace(0, 0.2, len(raw))
    raw[:, 6] = 1  # identity quaternion, xyzw
    raw[:, 7] = np.linspace(0, 0.4, len(raw))
    csv = out / "motion.csv"
    np.savetxt(csv, raw, delimiter=",")
    path = out / "reference.npz"
    convert_csv_to_npz(csv, path, input_fps=25, output_fps=50, model=model)
    store = ArtifactStore(out / "artifacts")
    robot = RobotSpec("synthetic-29", store.put_bytes(xml.encode(), "application/xml"),
                      G1_JOINT_NAMES, tuple(model.body(i).name for i in range(1, model.nbody)))
    ref = ReferenceCompiler(store, robot).import_npz(path)
    arrays = read_npz(store, ref.motion_npz)
    np.testing.assert_equal((ref.num_frames, ref.fps), (21, 50))
    np.testing.assert_allclose(arrays["body_pos_w"][:, 0, 0], np.linspace(0, .2, 21), atol=1e-6)
    np.testing.assert_allclose(arrays["joint_pos"][:, 0], np.linspace(0, .4, 21), atol=1e-6)
    print(f"PASS: {ref.num_frames} frames at {ref.fps:g} Hz, {model.nbody-1} bodies. {path}")


if __name__ == "__main__":
    main()

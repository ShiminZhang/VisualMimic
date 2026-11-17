"""Prepare edited/unedited G1 inputs from a source-only canonical motion manifest.

The manifest contains `source` (ArtifactRef) and `instruction`. Targets and dataset
labels are neither required nor loaded. Execute the resulting NPZ files with
visualmimic.execute_reference in the pinned SuperMimic environment.
"""
import argparse
from pathlib import Path
import json
import sys
import traceback
import numpy as np
from .contracts import ArtifactRef
from .storage import ArtifactStore, encode
from .robot_contracts import RobotSpec
from .retargeting import GMRRetargeter, RetargetConfig
from .backends.motionfix_editor import MotionFixEditor


def gmr_robot_spec(store):
    import mujoco
    from general_motion_retargeting import ROBOT_XML_DICT
    path = Path(ROBOT_XML_DICT['unitree_g1'])
    model = mujoco.MjModel.from_xml_path(str(path))
    joints = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt))
    bodies = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, j) for j in range(1, model.nbody))
    return RobotSpec('unitree-g1-29dof-gmr', store.put_bytes(path.read_bytes(), 'application/xml'), joints, bodies)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--source-artifacts', required=True, help='Only the source artifact directory; do not mount labels')
    p.add_argument('--motionfix-repository', required=True)
    p.add_argument('--gmr-repository', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--body-models', required=True)
    p.add_argument('--clip-config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    store = ArtifactStore(out / 'artifacts', allowed_read_roots=[args.source_artifacts])
    status = out / 'status.json'
    status.write_bytes(encode({'status': 'running', 'arguments': vars(args)}))
    try:
        manifest = json.loads(Path(args.input).read_text())
        if set(manifest) - {'source', 'instruction', 'sample_id'}:
            raise ValueError('Source manifest contains unsupported fields')
        source = store.read_motion(ArtifactRef(**manifest['source']))
        instruction = manifest['instruction']
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Nonempty instruction required')
        sys.path.insert(0, str(Path(args.gmr_repository).resolve()))
        robot = gmr_robot_spec(store)
        editor = MotionFixEditor(args.motionfix_repository, args.checkpoint, args.body_models,
                                 args.clip_config, device=args.device)
        options = editor.options(args.seed)
        edited = editor.edit(source, instruction, options)
        result = {'input': manifest, 'editor': editor.identity, 'generation': options,
                  'robot': robot, 'retarget_config': RetargetConfig(), 'branches': {}}
        for name, motion in [('source', source), ('edited', edited)]:
            human = store.put_motion(motion)
            retargeted = GMRRetargeter(store).retarget(motion, robot)
            if retargeted.status != 'accepted':
                raise ValueError(f'{name} retarget rejected: {retargeted.reasons}')
            r = retargeted.motion
            filename = out / f'{name}_robot.npz'
            np.savez_compressed(filename, timestamps_s=r.timestamps_s, root_pos_w=r.root_pos_w,
                                root_quat_wxyz=r.root_quat_wxyz, joint_pos=r.joint_pos,
                                valid_mask=r.valid_mask, joint_names=np.array(robot.joint_names))
            result['branches'][name] = {'human': human, 'robot_motion': store.put_bytes(filename.read_bytes(), 'application/x-npz'),
                                        'diagnostics': retargeted.diagnostics, 'mapping_id': retargeted.mapping_id}
        (out / 'result.json').write_bytes(encode(result))
        status.write_bytes(encode({'status': 'prepared', 'physics_execution': 'not_run'}))
        print(str(out / 'result.json'))
    except Exception as exc:
        status.write_bytes(encode({'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()}))
        raise


if __name__ == '__main__':
    main()

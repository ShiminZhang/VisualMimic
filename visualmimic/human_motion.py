"""Common 22-joint human pose boundary for MotionFix and video reconstructions."""
import numpy as np
from .contracts import MotionSpec, FieldSpec, MotionSequence

SMPL_BODY_NAMES = ('pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2',
                   'left_ankle','right_ankle','spine3','left_foot','right_foot','neck',
                   'left_collar','right_collar','head','left_shoulder','right_shoulder',
                   'left_elbow','right_elbow','left_wrist','right_wrist')
SMPL_BODY_PARENTS = (-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19)
HUMAN_SPEC = MotionSpec('smpl-body22-aa-zup-v1', 'smpl-local-axis-angle-and-world-joints',
                        'smpl-body22', 'right-handed-z-up', 'metres-radians',
                        {'rots': FieldSpec((22,3)), 'trans': FieldSpec((3,)),
                         'joint_positions': FieldSpec((22,3))})


def from_motionfix(raw, fps=30.):
    def array(v):
        if hasattr(v, 'detach'): v = v.detach().cpu().numpy()
        return np.asarray(v, dtype=np.float32)
    rots = array(raw['rots']).reshape(-1,22,3)
    return MotionSequence(HUMAN_SPEC, np.arange(len(rots)) / fps,
                          {'rots': rots, 'trans': array(raw['trans']),
                           'joint_positions': array(raw['joint_positions'])}, np.ones(len(rots), dtype=bool)).validate()


def world_joint_frames(motion):
    from scipy.spatial.transform import Rotation
    motion.validate()
    if motion.spec != HUMAN_SPEC or not motion.valid_mask.all():
        raise ValueError('Retargeting requires complete canonical human poses')
    local = motion.fields['rots']
    for t in range(len(local)):
        rotations = []
        for j, parent in enumerate(SMPL_BODY_PARENTS):
            r = Rotation.from_rotvec(local[t,j])
            rotations.append(r if parent < 0 else rotations[parent] * r)
        yield {name: (motion.fields['joint_positions'][t,j].copy(), rotations[j].as_quat(scalar_first=True))
               for j,name in enumerate(SMPL_BODY_NAMES)}

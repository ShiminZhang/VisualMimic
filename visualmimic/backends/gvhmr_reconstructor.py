"""Video-only GVHMR boundary with a fixed neutral SMPL-H editor representation."""
from pathlib import Path
from hashlib import file_digest
import subprocess
import tempfile
import numpy as np
from ..contracts import BackendIdentity, MotionSequence
from ..human_motion import HUMAN_SPEC
from ..reconstruction import ReconstructionResult
from ..storage import digest


def y_up_to_canonical(global_orient, body_pose, transl):
    from scipy.spatial.transform import Rotation
    root = np.asarray(global_orient).reshape(-1, 3)
    body = np.asarray(body_pose).reshape(-1, 21, 3)
    trans = np.asarray(transl).reshape(-1, 3)
    if len(root) < 2 or len(root) != len(body) or len(root) != len(trans):
        raise ValueError('Invalid GVHMR prediction lengths')
    if not all(np.isfinite(v).all() for v in (root, body, trans)):
        raise ValueError('Nonfinite GVHMR prediction')
    rotation = np.array([[1.,0,0],[0,0,-1.],[0,1.,0]])
    root_z = (Rotation.from_matrix(rotation) * Rotation.from_rotvec(root)).as_rotvec()
    rots = np.concatenate([root_z[:,None], body], axis=1).astype(np.float32)
    # Both paths use a fixed neutral body; subtracting initial translation removes
    # the constant template-pelvis pivot offset introduced by changing world axes.
    trans_z = ((trans - trans[:1]) @ rotation.T).astype(np.float32)
    return rots, trans_z


class GVHMRReconstructor:
    def __init__(self, store, repository, python, smplh_models, static_camera=True):
        self.store = store
        self.repository = Path(repository).resolve()
        # Resolving the venv's Python symlink selects the base interpreter and
        # loses its site-packages. Keep the invoked path inside the venv.
        self.python = str(Path(python).expanduser().absolute())
        self.smplh_models = Path(smplh_models).resolve()
        self.static_camera = static_camera
        assets = ['body_models/smplx/SMPLX_NEUTRAL.npz',
                  'gvhmr/gvhmr_siga24_release.ckpt', 'hmr2/epoch=10-step=25000.ckpt',
                  'vitpose/vitpose-h-multi-coco.pth', 'yolo/yolov8x.pt']
        hashes = {}
        for asset in assets:
            path = self.repository / 'inputs/checkpoints' / asset
            if not path.is_file():
                raise FileNotFoundError(f'GVHMR requires {path}')
            with path.open('rb') as stream:
                hashes[asset] = file_digest(stream, 'sha256').hexdigest()
        with (self.smplh_models / 'smplh/SMPLH_NEUTRAL.npz').open('rb') as stream:
            hashes['canonical_smplh'] = file_digest(stream, 'sha256').hexdigest()
        self.identity = BackendIdentity('gvhmr', 'ee960bb6e2ea2d381aa97f08e9b71ef320b624b1+headless-import',
            digest({'assets': hashes, 'static_camera': static_camera, 'fps':30,
                    'shape': 'neutral-smplh', 'translation': 'initial-origin', 'axes': 'y-to-z-up'}),
            hashes['gvhmr/gvhmr_siga24_release.ckpt'])

    def reconstruct(self, video):
        import cv2
        import torch
        import smplx
        from scipy.spatial.transform import Rotation
        t = np.asarray(video.timestamps_s)
        if len(t) < 2 or not np.allclose(np.diff(t), 1/30., atol=1e-6):
            raise ValueError('GVHMR requires explicit 30 fps video, without implicit frame-rate changes')
        content = self.store.read_bytes(video.video)
        with tempfile.TemporaryDirectory(prefix='gvhmr-', dir=self.store.root) as tmp:
            run = Path(tmp)
            input_path = run / 'input.mp4'
            input_path.write_bytes(content)
            cap = cv2.VideoCapture(str(input_path))
            try:
                observed = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                fps = cap.get(cv2.CAP_PROP_FPS)
            finally:
                cap.release()
            if observed != (len(t), video.width, video.height) or not np.isclose(fps, 30, atol=.01):
                raise ValueError('Encoded video disagrees with input manifest')
            command = [self.python, str(Path(__file__).with_name('gvhmr_worker.py')),
                       '--video', str(input_path), '--output_root', str(run / 'prediction')]
            if self.static_camera:
                command.append('--static_cam')
            completed = subprocess.run(command, cwd=self.repository, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            log = self.store.put_bytes(completed.stdout.encode(), 'text/plain')
            if completed.returncode:
                raise RuntimeError(f'GVHMR failed ({completed.returncode}); log: {log.uri}')
            path = run / 'prediction/input/global_params.npz'
            raw = self.store.put_bytes(path.read_bytes(), 'application/x-npz')
            with np.load(path, allow_pickle=False) as prediction:
                rots, trans = y_up_to_canonical(prediction['global_orient'], prediction['body_pose'], prediction['transl'])
            if len(rots) != len(t):
                raise ValueError('Reconstruction changed video frame count')
        # Rebuild fixed-neutral body joints in exactly the same representation as
        # the editor output; predicted shape remains in the raw diagnostic only.
        rotations = torch.from_numpy(Rotation.from_rotvec(rots.reshape(-1,3)).as_matrix().reshape(-1,22,3,3)).float()
        model = smplx.SMPLHLayer(str(self.smplh_models / 'smplh'), gender='neutral', ext='npz').eval()
        with torch.inference_mode():
            body = model(global_orient=rotations[:,:1], body_pose=rotations[:,1:], transl=torch.from_numpy(trans))
        motion = MotionSequence(HUMAN_SPEC, t.copy(), {'rots': rots, 'trans': trans,
            'joint_positions': body.joints[:,:22].numpy().astype(np.float32)}, np.ones(len(t), dtype=bool)).validate()
        diagnostics = self.store.put_json({'backend': self.identity, 'raw_prediction':raw, 'log':log,
            'input_video':video.video, 'static_camera':self.static_camera, 'shape':'fixed-neutral-smplh',
            'transform':'y-up to z-up; initial translation removed; no ground-truth alignment',
            'confidence':'not_calibrated'})
        return ReconstructionResult(self.store.put_motion(motion), None, diagnostics)

"""SMPL-H mesh video preparation using AITViewer's offscreen renderer."""
from pathlib import Path
from hashlib import file_digest
import tempfile
import sys
import numpy as np
from ..contracts import BackendIdentity
from ..human_motion import HUMAN_SPEC
from ..rendering import RenderResult
from ..storage import digest


class HumanMeshRenderer:
    def __init__(self, store, body_models):
        self.store = store
        self.body_models = Path(body_models).resolve()
        with (self.body_models / 'smplh/SMPLH_NEUTRAL.npz').open('rb') as stream:
            checksum = file_digest(stream, 'sha256').hexdigest()
        self.identity = BackendIdentity('aitviewer-smplh', '1.14.2',
                                        digest({'body_model': checksum, 'shape': 'neutral',
                                                'context': 'egl' if sys.platform.startswith('linux') else 'platform-default'}))

    def render(self, source, config):
        import torch
        import smplx
        import imageio.v2 as imageio
        from scipy.spatial.transform import Rotation
        from aitviewer.configuration import CONFIG
        from aitviewer.headless import HeadlessRenderer
        from aitviewer.renderables.meshes import Meshes
        source.validate()
        if source.spec != HUMAN_SPEC or not source.valid_mask.all():
            raise ValueError('Renderer requires complete canonical human motion')
        if len(source.timestamps_s) < 2 or not np.allclose(np.diff(source.timestamps_s), 1/config.fps, atol=1e-6):
            raise ValueError('Explicitly resample source before rendering')
        expected = ('fixed-clip-center', 'neutral-gray', 'aitviewer-default', 'white-floor')
        if (config.camera, config.appearance, config.lighting, config.background) != expected:
            raise ValueError(f'Unsupported render protocol, expected {expected}')
        model = smplx.SMPLHLayer(str(self.body_models / 'smplh'), gender='neutral', ext='npz').eval()
        rotations = torch.from_numpy(Rotation.from_rotvec(source.fields['rots'].reshape(-1,3)).as_matrix().reshape(-1,22,3,3)).float()
        with torch.inference_mode():
            mesh = model(global_orient=rotations[:,:1], body_pose=rotations[:,1:],
                         transl=torch.from_numpy(source.fields['trans']))
        vertices = mesh.vertices.numpy()
        # Explicit z-up -> renderer y-up. Geometry and camera share this frame.
        vertices = vertices[..., [0,2,1]].copy()
        vertices[...,2] *= -1
        floor_shift = -float(vertices[...,1].min())
        vertices[...,1] += floor_shift
        low, high = vertices.min((0,1)), vertices.max((0,1))
        target = (low + high) / 2
        radius = float(np.linalg.norm(high - low) / 2)
        # Fixed camera for the entire sequence, fitting its bounding sphere at 45deg.
        distance = max(3., radius / np.sin(np.deg2rad(22.5)) * 1.25 * max(1.,config.height/config.width))
        direction = np.array([1.,.35,1.]); direction /= np.linalg.norm(direction)
        position = target + direction * distance
        CONFIG.update_conf({'z_up':False, 'playback_fps': config.fps,
                            'auto_set_floor':False, 'auto_set_camera_target':False})
        if sys.platform.startswith('linux'):
            from moderngl_window.context.headless.window import Window
            from unittest.mock import patch
            # AITViewer 1.14.2 does not forward its **kwargs to the window.
            # PYOPENGL_PLATFORM does not select ModernGL's standalone backend.
            # Scope the resolver override to construction of this headless viewer.
            class EGLWindow(Window):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs, backend='egl')
            with patch('aitviewer.viewer.get_local_window_cls', return_value=EGLWindow):
                viewer = HeadlessRenderer(size=(config.width, config.height))
        else:
            viewer = HeadlessRenderer(size=(config.width, config.height))
        try:
            viewer.auto_set_floor = False
            viewer.auto_set_camera_target = False
            viewer.scene.add(Meshes(vertices, model.faces, color=(.65,.65,.65,1), z_up=False))
            viewer.scene.camera.position = position
            viewer.scene.camera.target = target
            viewer.scene.background_color = np.array([1.,1.,1.,1.])
            with tempfile.TemporaryDirectory(prefix='human-video-', dir=self.store.root) as tmp:
                path = Path(tmp) / 'source.mp4'
                with imageio.get_writer(str(path), fps=config.fps, codec='libx264', quality=8,
                                        macro_block_size=1) as writer:
                    for index in range(len(vertices)):
                        viewer.scene.current_frame_id = index
                        frame = np.asarray(viewer.get_frame())
                        if frame.shape[:2] != (config.height,config.width):
                            raise ValueError('Renderer output size mismatch')
                        writer.append_data(frame[:,:,:3])
                video = self.store.put_bytes(path.read_bytes(), 'video/mp4')
        finally:
            viewer.on_close()
        metadata = self.store.put_json({'backend':self.identity, 'config':config,
            'source_hash':digest(source), 'camera_position_yup':position, 'camera_target_yup':target,
            'floor_shift_y_m':floor_shift, 'frames':len(vertices),
            'scope':'preparation-only camera metadata; never pass this to video inference'})
        return RenderResult(video, tuple(float(v) for v in source.timestamps_s), digest(config), metadata)

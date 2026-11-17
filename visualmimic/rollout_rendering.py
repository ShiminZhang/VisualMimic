"""Render recorded robot states, without executing or resetting a policy."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .reference_compiler import read_npz
from .robot_contracts import validate_rollout_arrays


@dataclass(frozen=True)
class RobotRenderConfig:
    width: int = 640
    height: int = 480
    fps: int = 25
    azimuth: float = 135.
    elevation: float = -18.
    distance: float = 3.4
    camera: str = 'root-follow'


def render_rollout(rollout, store, output, config=RobotRenderConfig(), model_xml=None):
    import mujoco
    import imageio.v2 as imageio
    if model_xml is None:
        from supermimic.data.motion_convert import g1_xml_path
        model_xml = g1_xml_path()
    model_xml = Path(model_xml)
    from hashlib import sha256
    a = validate_rollout_arrays(read_npz(store, rollout.trajectory))
    meta = store.read_json(rollout.metadata)
    if sha256(model_xml.read_bytes()).hexdigest() != meta['robot']['model_asset']['sha256']:
        raise ValueError('Rendering model differs from execution model')
    if config.camera not in {'fixed-initial-root', 'root-follow'}:
        raise ValueError('Unknown camera protocol')
    dt = float(a['timestamps_s'][1] - a['timestamps_s'][0])
    stride = round(1 / (config.fps * dt))
    if stride < 1 or not np.isclose(stride * dt, 1 / config.fps):
        raise ValueError('Render fps must divide control rate')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    spec = mujoco.MjSpec.from_file(str(model_xml))
    spec.worldbody.add_geom(name='video_floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, .1], rgba=[.22, .25, .30, 1])
    spec.worldbody.add_light(pos=[0, -3, 5], dir=[0, 0, -1])
    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = config.width, config.height
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=config.height, width=config.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = config.azimuth, config.elevation, config.distance
    camera.lookat[:] = [*a['states'][0, :2], .65]
    option = mujoco.MjvOption()
    option.geomgroup[3:] = False
    indices = list(range(0, len(a['states']), stride))
    if indices[-1] != len(a['states']) - 1:
        indices.append(len(a['states']) - 1)
    writer = imageio.get_writer(str(output), fps=config.fps, codec='libx264', quality=8)
    try:
        for index in indices:
            if config.camera == 'root-follow':
                camera.lookat[:2] = a['states'][index, :2]
            data.qpos[:] = a['states'][index, :36]
            data.qvel[:] = a['states'][index, 36:]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            writer.append_data(renderer.render().copy())
    finally:
        writer.close()
        renderer.close()
    video = store.put_bytes(output.read_bytes(), 'video/mp4')
    return {'video': video, 'config': config, 'state_indices': indices,
            'source_rollout': rollout.rollout_id, 'outcome': rollout.outcome,
            'frame_count': len(indices), 'terminal_frame_included': True,
            'timing': 'uniform decimation; final state appended if off-grid, no loop or hold'}

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from visualmimic.backends.gvhmr_reconstructor import y_up_to_canonical


def test_axis_conversion_rotates_root_and_trajectory_but_preserves_local_pose():
    root = np.array([[.3,.1,.2],[.1,.4,-.2]])
    body = np.arange(126).reshape(2,21,3) / 1000
    trans = np.array([[2.,3.,4.],[3.,5.,7.]])
    rots, out = y_up_to_canonical(root, body, trans)
    conversion = np.array([[1.,0,0],[0,0,-1.],[0,1.,0]])
    np.testing.assert_allclose(Rotation.from_rotvec(rots[:,0]).as_matrix(),
                               conversion @ Rotation.from_rotvec(root).as_matrix(), atol=1e-7)
    np.testing.assert_allclose(rots[:,1:], body, atol=1e-8)
    np.testing.assert_allclose(out, [[0,0,0],[1,-3,2]])
    _, moved = y_up_to_canonical(root, body, trans + [7,11,-3])
    np.testing.assert_allclose(out, moved)


def test_reconstruction_rejects_missing_or_nonfinite_frames():
    with pytest.raises(ValueError, match='lengths'):
        y_up_to_canonical(np.zeros((2,3)), np.zeros((3,63)), np.zeros((2,3)))
    with pytest.raises(ValueError, match='Nonfinite'):
        y_up_to_canonical(np.full((2,3),np.nan), np.zeros((2,63)), np.zeros((2,3)))


def test_reconstructor_preserves_virtual_environment_python(tmp_path):
    import sys
    from pathlib import Path
    from visualmimic.backends.gvhmr_reconstructor import GVHMRReconstructor
    from visualmimic.storage import ArtifactStore
    python = tmp_path / 'venv/bin/python'
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    repo = tmp_path / 'repo'
    # Asset placeholders only exercise preflight identity, never inference.
    for name in ['body_models/smplx/SMPLX_NEUTRAL.npz',
                 'gvhmr/gvhmr_siga24_release.ckpt', 'hmr2/epoch=10-step=25000.ckpt',
                 'vitpose/vitpose-h-multi-coco.pth', 'yolo/yolov8x.pt']:
        p = repo / 'inputs/checkpoints' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'identity-only-test-placeholder')
    body = tmp_path / 'body/smplh/SMPLH_NEUTRAL.npz'
    body.parent.mkdir(parents=True)
    body.write_bytes(b'identity-only-test-placeholder')
    backend = GVHMRReconstructor(ArtifactStore(tmp_path/'artifacts'), repo,
                                 python, tmp_path/'body')
    assert Path(backend.python) == python
    assert Path(backend.python) != python.resolve()

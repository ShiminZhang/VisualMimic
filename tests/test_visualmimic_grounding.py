"""Ground placement must preserve vertical motion and remove arbitrary height origins."""
import numpy as np
import pytest
mujoco = pytest.importorskip('mujoco')
from visualmimic.retargeting import ground_foot_spheres


def test_constant_ground_shift_preserves_jump_and_is_origin_invariant():
    spheres = ''.join(f'<geom type="sphere" size=".01" pos="{x} {y} -.5"/>'
                      for x, y in ((.1,.1),(.1,-.1),(-.1,.1),(-.1,-.1)))
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody><body><freejoint/>
      <body name="left_ankle_roll_link">{spheres}</body>
      <body name="right_ankle_roll_link">{spheres}</body>
      </body></worldbody></mujoco>''')
    a = np.zeros((3,7)); a[:,3] = 1; a[:,2] = [-.2,.3,-.1]
    b = a.copy(); b[:,2] += 13
    before = np.diff(a[:,2]).copy()
    ground_foot_spheres(model, a)
    ground_foot_spheres(model, b)
    np.testing.assert_allclose(a, b, atol=1e-12)
    np.testing.assert_allclose(np.diff(a[:,2]), before)
    assert a[:,2].min() == pytest.approx(.512)

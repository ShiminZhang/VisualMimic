import numpy as np
from visualmimic.termination_diagnostics import predicates


def test_height_boundary_and_tilt_is_yaw_invariant():
    reference = np.zeros((3, 36))
    reference[:, 3] = 1
    states = reference.copy()
    states[:, 2] = [0.25, 0.251, 0]
    # A 180-degree yaw leaves gravity unchanged; a 90-degree roll does not.
    states[1, 3:7] = [0, 0, 0, 1]
    states[2, 3:7] = [np.sqrt(0.5), np.sqrt(0.5), 0, 0]
    height, tilt, bad_height, bad_tilt = predicates(states, reference, 0.25, 0.8)
    np.testing.assert_allclose(height, [0.25, 0.251, 0])
    np.testing.assert_allclose(tilt, [0, 0, 1], atol=1e-12)
    np.testing.assert_array_equal(bad_height, [False, True, False])
    np.testing.assert_array_equal(bad_tilt, [False, False, True])

import numpy as np
import pytest

from rotml.physics.rotor import (
    INERTIA_TO_MHZ,
    inertial_defect,
    is_linear,
    masses_for,
    principal_moments,
    ray_asymmetry,
    rotational_constants,
)


def test_diatomic_matches_textbook_formula():
    # CO with r = 1.128 A: I = mu r^2, B = 505379.07 / I MHz.
    m = masses_for(["C", "O"])
    r = 1.128
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]])
    mu = m[0] * m[1] / m.sum()
    expected_B = INERTIA_TO_MHZ / (mu * r**2) / 1000.0
    A, B, C = rotational_constants(coords, m)
    assert A == 0.0  # linear convention
    assert B == pytest.approx(expected_B, rel=1e-10)
    assert C == pytest.approx(expected_B, rel=1e-10)
    assert is_linear(A, B)


def test_methane_qm9_reference():
    # gdb_1 geometry from QM9; reference constants 157.7118, 157.70997, 157.70699 GHz.
    coords = np.array(
        [
            [-0.0127, 1.0858, 0.0080],
            [0.0022, -0.0060, 0.0020],
            [1.0117, 1.4638, 0.0003],
            [-0.5408, 1.4475, -0.8766],
            [-0.5238, 1.4379, 0.9064],
        ]
    )
    consts = rotational_constants(coords, masses_for(["C", "H", "H", "H", "H"]))
    assert consts == pytest.approx([157.7118, 157.70997, 157.70699], rel=2e-4)


def test_invariant_to_rotation_and_translation():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(6, 3))
    masses = masses_for(["C", "C", "O", "H", "H", "N"])
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    moved = coords @ q.T + np.array([3.0, -1.0, 7.0])
    assert principal_moments(moved, masses) == pytest.approx(
        principal_moments(coords, masses), rel=1e-10
    )


def test_planar_molecule_has_zero_inertial_defect():
    # Rigid planar geometry: I_c = I_a + I_b exactly.
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.3, 0.0], [-0.4, 1.1, 0.0], [0.7, -0.9, 0.0]])
    A, B, C = rotational_constants(coords, masses_for(["C", "O", "H", "N"]))
    assert inertial_defect(A, B, C) == pytest.approx(0.0, abs=1e-8)


def test_ray_asymmetry_limits():
    assert ray_asymmetry(10.0, 2.0, 2.0) == pytest.approx(-1.0)  # prolate
    assert ray_asymmetry(10.0, 10.0, 2.0) == pytest.approx(1.0)  # oblate

import numpy as np

from rotml.analysis.identifiability import count_matches


def brute_force(A, B, C, delta, groups=None):
    n = len(A)
    out = np.zeros(n, dtype=int)
    r = np.log1p(delta)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if groups is not None and groups[i] != groups[j]:
                continue
            lin_i, lin_j = A[i] == 0, A[j] == 0
            if lin_i != lin_j:
                continue
            if lin_i:
                ok = abs(np.log(B[i]) - np.log(B[j])) <= r
            else:
                ok = all(
                    abs(np.log(x[i]) - np.log(x[j])) <= r for x in (A, B, C)
                )
            out[i] += ok
    return out


def test_matches_brute_force():
    rng = np.random.default_rng(1)
    n = 300
    C = rng.uniform(1.0, 1.1, n)
    B = C * rng.uniform(1.0, 1.05, n)
    A = B * rng.uniform(1.0, 1.3, n)
    A[:5] = 0.0  # a few linear rotors
    B[:5] = C[:5]
    groups = rng.integers(0, 3, n)
    for delta in (0.005, 0.02):
        assert np.array_equal(count_matches(A, B, C, delta), brute_force(A, B, C, delta))
        assert np.array_equal(
            count_matches(A, B, C, delta, groups=groups),
            brute_force(A, B, C, delta, groups=groups),
        )


def test_extra_observable_only_removes_matches():
    rng = np.random.default_rng(2)
    n = 200
    C = rng.uniform(1.0, 1.02, n)
    B, A = C * 1.01, C * 1.2
    mu = rng.uniform(0, 4, n)
    base = count_matches(A, B, C, 0.02)
    with_mu = count_matches(A, B, C, 0.02, extra={"mu": (mu, 0.2)})
    assert np.all(with_mu <= base)
    assert with_mu.sum() < base.sum()

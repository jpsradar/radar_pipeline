"""
validation/golden_tests/test_cfar_homogeneous.py

Golden test: CA-CFAR maintains the requested Pfa under homogeneous exponential clutter/noise.

Goal
----
Guarantee that CA-CFAR threshold scaling remains correct when:
- refactoring detection modules
- changing numpy/scipy versions
- integrating CFAR into a larger pipeline

Test strategy
-------------
We generate i.i.d. exponential power samples for:
- CUT: x_cut ~ Exp(mean=mu)
- References: x_ref[i] ~ Exp(mean=mu), i=1..Nref

Under the exponential assumption, CA-CFAR should achieve:
    P(x_cut > alpha * mean(x_ref)) = Pfa

We estimate the empirical false alarm rate over many independent trials and check
it is close to Pfa within statistical tolerance.

How to run
----------
    pytest -q

Notes
-----
- This is a statistical test; we control randomness with a fixed seed.
- Tolerance is set based on binomial standard deviation.
"""

from __future__ import annotations

import math
import numpy as np

from core.detection.cfar import ca_cfar_alpha


def test_ca_cfar_empirical_pfa_matches_target() -> None:
    """
    Empirical Pfa under homogeneous exponential noise should match target Pfa.

    We use independent trials (no sliding window correlation) to test the core math.
    """
    rng = np.random.default_rng(12345)

    pfa_target = 1e-6
    n_ref = 32  # reference cells
    mu = 1.0    # mean of exponential distribution (scale parameter)

    # Number of independent trials.
    # For pfa=1e-6 we need many trials to see enough false alarms; however, we keep this
    # test runtime reasonable by using a moderate pfa for the golden test.
    #
    # IMPORTANT: We *do not* use pfa_target=1e-6 here because we'd need ~1e8 trials.
    # Instead, we validate the *formula* at a practical pfa, which locks the alpha relation.
    pfa_test = 1e-3

    alpha = ca_cfar_alpha(pfa=pfa_test, n_ref=n_ref)

    n_trials = 2_000_000  # should run fast with vectorization
    # Generate CUT and reference powers (exponential => gamma(k=1, theta=mu))
    x_cut = rng.exponential(scale=mu, size=n_trials)
    x_ref = rng.exponential(scale=mu, size=(n_trials, n_ref))

    z = np.mean(x_ref, axis=1)
    thr = alpha * z

    fa = np.mean(x_cut > thr)  # empirical false alarm rate

    # Binomial standard deviation for estimated proportion:
    # std = sqrt(p*(1-p)/n). We use 6-sigma bound for robustness across environments.
    std = math.sqrt(pfa_test * (1.0 - pfa_test) / n_trials)
    tol = 6.0 * std

    assert abs(fa - pfa_test) < tol, f"Empirical Pfa={fa} differs from target={pfa_test} by > {tol}"


def test_ca_cfar_alpha_monotonic_in_pfa() -> None:
    """
    Sanity: as desired Pfa decreases, alpha must increase (more conservative threshold).
    """
    n_ref = 32
    a1 = ca_cfar_alpha(pfa=1e-2, n_ref=n_ref)
    a2 = ca_cfar_alpha(pfa=1e-3, n_ref=n_ref)
    a3 = ca_cfar_alpha(pfa=1e-4, n_ref=n_ref)

    assert a1 < a2 < a3
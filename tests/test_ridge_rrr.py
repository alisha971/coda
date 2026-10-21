"""Layer-1 estimators against independent reference calculations.

The fixed-Z Schur-complement solver must equal a direct solve of the normal
equations and scikit-learn's Ridge; λ selection must never read evaluation
rows; reduced-rank ridge must recover a planted low-rank coupling.
"""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.evaluation.metrics import (
    aggregate,
    block_partial_r2,
    gaussian_loglik,
    interval_coverage,
    loglik_increment,
)
from src.models.ridge import RidgeEngine, ZCache, block_col_groups, grouped_folds
from src.models.rrr import fit_rrr, leading_strength
from src.models.variance import fit_variance

BLOCKS = block_col_groups([2, 1, 2, 3])


def synthetic(n_sessions=60, per=40, p=12, q=8, m=8, coupling=0.0, seed=0, rank=None):
    rng = np.random.default_rng(seed)
    n = n_sessions * per
    groups = np.repeat(np.arange(n_sessions), per)
    Z = rng.normal(size=(n, p))
    A = rng.normal(size=(n, q)) + 0.3 * Z[:, :q]          # A correlated with Z
    Bz = rng.normal(size=(p, m)) * 0.3
    if rank is None:
        Ba = np.eye(q, m) * coupling
    else:
        Ba = (rng.normal(size=(q, rank)) @ rng.normal(size=(rank, m))) * coupling
    u = rng.normal(size=(n_sessions, m))[groups] * 0.5      # session-level clustering
    Y = Z @ Bz + A @ Ba + u + rng.normal(size=(n, m))
    test = groups >= int(0.8 * n_sessions)
    return Z, A, Y, groups, ~test, test


def standardised(X, rows):
    mu, sd = X[rows].mean(0), X[rows].std(0)
    return (X - mu) / np.where(sd > 0, sd, 1)


def direct_ridge(Z, A, Y, fit, lam):
    X = np.hstack([standardised(Z, fit), standardised(A, fit)]) if A is not None \
        else standardised(Z, fit)
    Xf = X[fit]
    ym = Y[fit].mean(0)
    beta = np.linalg.solve(Xf.T @ Xf + lam * np.eye(X.shape[1]), Xf.T @ (Y[fit] - ym))
    return X, beta, ym


def test_schur_solver_equals_a_direct_solve_and_sklearn():
    Z, A, Y, g, fit, test = synthetic(coupling=0.3)
    zc = ZCache(Z, g, fit, test, n_folds=5, seed=1)
    fz = zc.folds(Y)[-1]
    eng = RidgeEngine(zc, Y, BLOCKS)
    ap = eng._a_pieces(fz, A)
    lam = 37.0
    gZ, bA, _ = eng._solve(fz, ap, lam)
    pred = eng._predict(fz, ap, gZ, bA)
    X, beta, ym = direct_ridge(Z, A, Y, fit, lam)
    np.testing.assert_allclose(pred, X[test] @ beta + ym, rtol=1e-8, atol=1e-8)
    sk = Ridge(alpha=lam, fit_intercept=True).fit(X[fit], Y[fit])
    np.testing.assert_allclose(pred, sk.predict(X[test]), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(bA, beta[Z.shape[1]:], rtol=1e-8, atol=1e-10)


def test_z_only_solution_matches_a_direct_solve():
    Z, A, Y, g, fit, test = synthetic()
    zc = ZCache(Z, g, fit, test)
    eng = RidgeEngine(zc, Y, BLOCKS)
    fz = eng.folds[-1]
    gZ, _, _ = eng._solve(fz, None, 5.0)
    X, beta, ym = direct_ridge(Z, None, Y, fit, 5.0)
    np.testing.assert_allclose(eng._predict(fz, None, gZ, None), X[test] @ beta + ym,
                               rtol=1e-8, atol=1e-8)


def test_effective_df_equals_the_trace_of_the_hat_matrix():
    Z, A, Y, g, fit, test = synthetic(n_sessions=30)
    zc = ZCache(Z, g, fit, test)
    eng = RidgeEngine(zc, Y, BLOCKS, lambda_rel_grid=(0.01,))
    res = eng.run(A, return_edf=True)
    X, _, _ = direct_ridge(Z, A, Y, fit, 0)
    Xf = X[fit]
    lam = res.lambdas[0]
    H = Xf @ np.linalg.solve(Xf.T @ Xf + lam * np.eye(X.shape[1]), Xf.T)
    assert res.edf[0] == pytest.approx(np.trace(H), rel=1e-8)


def test_inner_folds_keep_sessions_whole():
    g = np.repeat(np.arange(23), 7)
    f = grouped_folds(g, 5, seed=3)
    for s in range(23):
        assert len(set(f[g == s])) == 1
    assert len(set(f)) == 5


def test_lambda_selection_never_reads_evaluation_rows():
    """Sentinel test targets: anything but identical λ means the test set leaked."""
    Z, A, Y, g, fit, test = synthetic(coupling=0.2)
    zc = ZCache(Z, g, fit, test)
    a = RidgeEngine(zc, Y, BLOCKS).run(A)
    Y2 = Y.copy()
    Y2[test] = 1e6 * np.random.default_rng(9).normal(size=Y2[test].shape)
    b = RidgeEngine(zc, Y2, BLOCKS).run(A)
    np.testing.assert_array_equal(a.lambdas, b.lambdas)
    np.testing.assert_allclose(a.pred_eval, b.pred_eval)


def test_fitting_and_evaluation_sessions_must_not_overlap():
    Z, A, Y, g, fit, test = synthetic()
    bad = test.copy()
    bad[np.flatnonzero(fit)[0]] = True
    with pytest.raises(ValueError):
        ZCache(Z, g, fit, bad)


def test_partial_r2_is_positive_with_coupling_and_near_zero_without():
    for coupling, positive in ((0.4, True), (0.0, False)):
        Z, A, Y, g, fit, test = synthetic(coupling=coupling, seed=4)
        zc = ZCache(Z, g, fit, test)
        eng = RidgeEngine(zc, Y, BLOCKS)
        full, red = eng.run(A), eng.run(None)
        pr2 = block_partial_r2(Y[test], full.pred_eval, red.pred_eval, BLOCKS)
        if positive:
            assert aggregate(pr2) > 0.05
        else:
            assert abs(aggregate(pr2)) < 0.02


def test_partial_r2_by_hand():
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    full = np.zeros_like(y) + 0.5
    red = np.zeros_like(y)
    # SSE full = 4 * 0.25 = 1; SSE reduced = 2  -> 1 - 1/2
    assert block_partial_r2(y, full, red, [np.array([0, 1])])[0] == pytest.approx(0.5)


def test_stacked_replicates_give_the_same_answer_as_separate_fits():
    """J3 stacks many synthetic targets as columns; each must be fitted as if alone."""
    Z, A, Y, g, fit, test = synthetic(coupling=0.2, seed=6)
    # Pure noise in the second replicate forces heavier shrinkage, so the column
    # groups choose different λ (regression test).
    Y2 = 5 * np.random.default_rng(1).normal(size=Y.shape)
    zc = ZCache(Z, g, fit, test)
    stacked = RidgeEngine(zc, np.hstack([Y, Y2]), block_col_groups([2, 1, 2, 3], 2)).run(A)
    assert len(set(stacked.lambda_rel)) > 1
    one = RidgeEngine(zc, Y, BLOCKS).run(A)
    two = RidgeEngine(zc, Y2, BLOCKS).run(A)
    np.testing.assert_allclose(stacked.pred_eval, np.hstack([one.pred_eval, two.pred_eval]))


# -- reduced-rank ridge -----------------------------------------------------


def test_rank_8_rrr_equals_full_ridge_at_the_same_lambda():
    Z, A, Y, g, fit, test = synthetic(coupling=0.3, seed=7)
    zc = ZCache(Z, g, fit, test)
    eng = RidgeEngine(zc, Y, BLOCKS)
    rr = fit_rrr(eng, A, BLOCKS, fixed=(0.01, 8))
    fz = eng.folds[-1]
    ap = eng._a_pieces(fz, A)
    gZ, bA, _ = eng._solve(fz, ap, 0.01 * len(fz.fit))
    np.testing.assert_allclose(rr.pred_eval, eng._predict(fz, ap, gZ, bA), atol=1e-8)


def test_rrr_recovers_a_planted_rank_one_coupling():
    Z, A, Y, g, fit, test = synthetic(n_sessions=120, coupling=0.6, rank=1, seed=8)
    zc = ZCache(Z, g, fit, test)
    rr = fit_rrr(RidgeEngine(zc, Y, BLOCKS), A, BLOCKS)
    assert rr.rank <= 2
    assert rr.shares[0] > 0.9


def test_leading_strength_is_larger_with_real_than_with_shuffled_partner():
    Z, A, Y, g, fit, test = synthetic(n_sessions=80, coupling=0.4, rank=1, seed=9)
    zc = ZCache(Z, g, fit, test)
    eng = RidgeEngine(zc, Y, BLOCKS)
    real = leading_strength(eng, A, 0.01)
    shuffled = leading_strength(eng, A[np.random.default_rng(0).permutation(len(A))], 0.01)
    assert real > 2 * shuffled


# -- variance model and log-likelihood --------------------------------------


def test_variance_model_tracks_a_known_variance_function():
    rng = np.random.default_rng(10)
    n_s, per = 80, 40
    g = np.repeat(np.arange(n_s), per)
    Z = rng.normal(size=(n_s * per, 3))
    sd = np.exp(0.5 * Z[:, 0])
    Y = (Z @ np.ones((3, 1))) + sd[:, None] * rng.normal(size=(n_s * per, 1))
    test = g >= 64
    zc = ZCache(Z, g, ~test, test)
    eng = RidgeEngine(zc, Y, [np.array([0])])
    res = eng.run(None, return_oof=True)
    oof_resid = Y[~test] - res.oof_fit
    vf = fit_variance(zc, oof_resid, None, eng.grid_rel)
    true_var = sd[test] ** 2
    assert np.corrcoef(vf.var_eval[:, 0], true_var)[0, 1] > 0.6
    assert (vf.var_eval > 0).all()


def test_loglik_and_coverage_by_hand():
    y = np.zeros((4, 1))
    assert gaussian_loglik(y, y, np.ones_like(y))[0, 0] == pytest.approx(-0.5 * np.log(2 * np.pi))
    inc = loglik_increment(y, y, np.ones_like(y), y + 1, np.ones_like(y), [np.array([0])])
    assert inc[0] == pytest.approx(0.5)
    rng = np.random.default_rng(0)
    yy = rng.normal(size=(20000, 1))
    cov = interval_coverage(yy, np.zeros_like(yy), np.ones_like(yy), [np.array([0])])
    assert cov[0.68][0] == pytest.approx(0.68, abs=0.01)
    assert cov[0.95][0] == pytest.approx(0.95, abs=0.01)


def test_cross_product_sse_equals_prediction_sse_and_picks_the_same_lambda():
    """The fast path must give the same numbers as forming predictions."""
    Z, A, Y, g, fit, test = synthetic(coupling=0.3, seed=11)
    Y2 = np.hstack([Y, 3 * np.random.default_rng(2).normal(size=Y.shape)])
    zc = ZCache(Z, g, fit, test)
    eng = RidgeEngine(zc, Y2, block_col_groups([2, 1, 2, 3], 2))
    for a in (None, A):
        slow = eng.run(a, return_oof=True)                 # prediction-based selection
        fast = eng.run(a, eval_sse_only=True)              # cross-product selection and SSE
        np.testing.assert_array_equal(slow.lambda_rel, fast.lambda_rel)
        np.testing.assert_allclose(fast.cv_sse, slow.cv_sse, rtol=1e-9)
        direct = ((Y2[test] - slow.pred_eval) ** 2).sum(axis=0)
        np.testing.assert_allclose(fast.extra["eval_sse"], direct, rtol=1e-9)

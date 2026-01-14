import numpy as np
import pytest

from fedchem.metrics import metrics


def test_rmsep_matches_manual():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 3.0, 2.0])
    expected = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    assert metrics.rmsep(y_true, y_pred) == pytest.approx(expected)


def test_rmsep_shape_mismatch_raises():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0])
    with pytest.raises(ValueError):
        metrics.rmsep(y_true, y_pred)


def test_cvrmsep_normalization_and_guard():
    y_true = np.array([2.0, -2.0, 4.0])
    y_pred = np.array([1.0, -1.0, 4.0])
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = float(np.mean(np.abs(y_true)))
    assert metrics.cvrmsep(y_true, y_pred) == pytest.approx(rmse / denom)
    with pytest.raises(ValueError):
        metrics.cvrmsep(np.zeros((2,)), np.zeros((2, 1)))
    with pytest.raises(ValueError):
        metrics.cvrmsep(np.zeros(3), np.zeros(3))


def test_mae_and_shape_guard():
    y_true = np.array([1.0, -1.0, 2.0])
    y_pred = np.array([0.0, -2.0, 2.0])
    expected = float(np.mean(np.abs(y_true - y_pred)))
    assert metrics.mae(y_true, y_pred) == pytest.approx(expected)
    with pytest.raises(ValueError):
        metrics.mae(np.zeros((2,)), np.zeros((2, 1)))


def test_r2_perfect_and_mismatch():
    y_true = np.array([1.0, 2.0, 3.0])
    assert metrics.r2(y_true, y_true) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        metrics.r2(np.zeros((2,)), np.zeros((2, 1)))


def test_coverage_and_interval_width():
    y_true = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.0, 2.0, 2.5])
    hi = np.array([1.5, 3.0, 4.0])
    assert metrics.coverage(y_true, lo, hi) == pytest.approx(1.0)
    assert metrics.interval_width(lo, hi) == pytest.approx(float(np.mean(hi - lo)))
    with pytest.raises(ValueError):
        metrics.coverage(y_true, lo, np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        metrics.interval_width(lo, np.array([1.0, 2.0]))

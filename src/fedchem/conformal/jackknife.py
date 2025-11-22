from typing import Tuple
import numpy as np
from tqdm import trange


def jackknife_plus_intervals(model_factory, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, alpha: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Compute jackknife+ prediction intervals.

    model_factory: callable that returns a fresh model instance with fit(X,y) and predict(X).
    X,y: calibration dataset (paired)
    X_test: test inputs to compute intervals for
    Returns (lower, upper) arrays of shape (n_test,)
    """
    n = X.shape[0]
    n_test = X_test.shape[0]
    # Collect leave-one-out predictions: for each i, train on all except i, predict on left-out and on X_test
    preds_test = np.zeros((n, n_test), dtype=float)
    preds_loo = np.zeros(n, dtype=float)
    for i in range(n):
        # prepare train set
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr, y_tr = X[mask], y[mask]
        m = model_factory()
        m.fit(X_tr, y_tr)
        preds_test[i] = np.asarray(m.predict(X_test)).ravel()
        # predict on left-out
        preds_loo[i] = float(np.asarray(m.predict(X[i:i+1])).ravel()[0])

    # For each test point, compute residual-like values and quantile
    q_low = np.zeros(n_test, dtype=float)
    q_high = np.zeros(n_test, dtype=float)
    for j in range(n_test):
        # Construct scores s_i_j = preds_test[i,j] - y_i (leave-one-out target)
        scores = preds_test[:, j] - preds_loo
        # Lower bound uses (1 - alpha)/2 quantile of scores
        q_low[j] = np.quantile(scores, alpha / 2.0)
        q_high[j] = np.quantile(scores, 1.0 - alpha / 2.0)

    # Intervals: for prediction point j, lower = preds_full[j] - q_high[j], upper = preds_full[j] - q_low[j]
    # Fit model on full data to get final predictions
    m_full = model_factory()
    m_full.fit(X, y)
    preds_full = np.asarray(m_full.predict(X_test)).ravel()
    lower = preds_full - q_high
    upper = preds_full - q_low
    return lower, upper

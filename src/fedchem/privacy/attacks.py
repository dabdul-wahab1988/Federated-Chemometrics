from typing import Tuple, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def exfiltration_reconstruction_error(X_original: np.ndarray, X_transformed: np.ndarray, T_matrix: np.ndarray) -> float:
    """Estimate normalized reconstruction error using pseudo-inverse of T.

    Returns ||X_original - X_trans_recon||_F / ||X_original||_F
    """
    try:
        T_pinv = np.linalg.pinv(T_matrix)
        X_recon = X_transformed @ T_pinv
        num = float(np.linalg.norm(X_original - X_recon, ord='fro'))
        den = float(np.linalg.norm(X_original, ord='fro'))
        return num / (den + 1e-12)
    except Exception:
        return float('inf')


def membership_inference_auc(X_in: np.ndarray,
                             X_out: np.ndarray,
                             features_fn=None,
                             classifier: Optional[str] = 'logreg',
                             test_size: float = 0.3,
                             random_state: int = 0) -> float:
    """Train a classifier to distinguish in/out samples and return AUC.

    features_fn: optional callable(X)->features; if None use identity flattening.
    classifier: 'logreg' or 'rf'
    """
    X_pos = np.asarray(X_in)
    X_neg = np.asarray(X_out)
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])])
    if features_fn is None:
        # flatten spectral features
        Xf = X.reshape(X.shape[0], -1)
    else:
        Xf = features_fn(X)

    X_tr, X_te, y_tr, y_te = train_test_split(Xf, y, test_size=test_size, random_state=random_state, stratify=y)
    if classifier == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)
    y_score = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_score)
    return float(auc)

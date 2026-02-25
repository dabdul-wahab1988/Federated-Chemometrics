"""
Outlier Diagnostic Script.

Identifies the 2 outliers mentioned by the reviewer using:
1. PCA + Hotelling's T2 (Spectral Outliers)
2. PLS Residuals (Property Outliers)
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

def analyze_outliers(path, name):
    df = pd.read_csv(path)
    y = df['Protein'].to_numpy().astype(float)
    spec_cols = [c for c in df.columns if c not in ['ID', 'Protein']]
    X = df[spec_cols].to_numpy().astype(float)
    
    # 1. Spectral Outliers (PCA)
    pca = PCA(n_components=5)
    X_scores = pca.fit_transform(X)
    # Calculate T2 (simplified)
    t2 = np.sum((X_scores / np.std(X_scores, axis=0))**2, axis=1)
    t2_idx = np.argmax(t2)
    
    # 2. Property/Model Outliers (PLS Residuals)
    pls = PLSRegression(n_components=5)
    pls.fit(X, y)
    y_hat = pls.predict(X).ravel()
    res = np.abs(y - y_hat)
    res_idx = np.argmax(res)
    
    print(f"\n--- Analysis for {name} ---")
    print(f"Max Spectral Outlier (T2) index: {t2_idx}, ID: {df.iloc[t2_idx]['ID']}")
    print(f"Max Residual Outlier index: {res_idx}, ID: {df.iloc[res_idx]['ID']}, Error: {res[res_idx]:.4f}")
    
    return res_idx

analyze_outliers("data/MA_A2.csv", "MA_A2")
analyze_outliers("data/MB_B2.csv", "MB_B2")

import numpy as np
import pytest

from fedchem.ct.pds_transfer import PDSTransfer


def make_linear_pair(n=40, d=6, bias=0.0, seed=0, matrix=None):
    rng = np.random.default_rng(seed)
    x_target = rng.normal(size=(n, d))
    if matrix is None:
        matrix = rng.normal(size=(d, d))
    x_source = x_target @ matrix + bias
    return x_source, x_target


def make_block_diag_matrix(d, block, seed=0):
    rng = np.random.default_rng(seed)
    mat = np.zeros((d, d))
    for start in range(0, d, block):
        stop = min(d, start + block)
        mat[start:stop, start:stop] = rng.normal(size=(stop - start, stop - start))
    return mat


def test_pds_transfer_unfitted_errors():
    p = PDSTransfer()
    assert p.estimated_bytes() == 0
    assert p.get_blocks() is None
    assert p.get_global_TC() is None
    assert p.to_dict()["diagnostics"] is None
    with pytest.raises(RuntimeError):
        _ = p.diagnostics
    with pytest.raises(RuntimeError):
        p.transform(np.zeros((2, 2)))


def test_pds_transfer_fit_shape_mismatch():
    p = PDSTransfer()
    with pytest.raises(ValueError):
        p.fit(np.zeros((2, 2)), np.zeros((2, 3)))


def test_pds_transfer_global_missing_mapping_raises():
    p = PDSTransfer()
    p._mode = "global"
    p._global_TC = None
    with pytest.raises(RuntimeError):
        p.transform(np.zeros((2, 2)))


def test_pds_transfer_clip_feature_noop_target_none():
    mat = make_block_diag_matrix(d=6, block=3, seed=7)
    x_source, x_target = make_linear_pair(bias=0.1, seed=8, matrix=mat)
    p = PDSTransfer(window=3, overlap=0, ridge=1e-12, use_global_affine=False)
    p.fit(x_source, x_target, clip_feature=1e6, clip_target=None)
    assert p.to_dict()["diagnostics"] is not None


def test_pds_transfer_clip_target_only_noop():
    mat = make_block_diag_matrix(d=6, block=3, seed=9)
    x_source, x_target = make_linear_pair(bias=0.1, seed=10, matrix=mat)
    p = PDSTransfer(window=3, overlap=0, ridge=1e-12, use_global_affine=False)
    p.fit(x_source, x_target, clip_feature=None, clip_target=1e6)
    assert p.to_dict()["diagnostics"] is not None


def test_pds_transfer_zero_features():
    x_source = np.zeros((3, 0))
    x_target = np.zeros((3, 0))
    p = PDSTransfer(window=3, overlap=0, ridge=1e-12, use_global_affine=False)
    p.fit(x_source, x_target)
    assert p.get_blocks() == []
    assert p.diagnostics.mean_rmse == pytest.approx(0.0)


def test_pds_transfer_blocks_mode():
    block = 3
    mat = make_block_diag_matrix(d=6, block=block, seed=4)
    x_source, x_target = make_linear_pair(bias=0.5, seed=1, matrix=mat)
    p = PDSTransfer(window=block, overlap=0, ridge=1e-12, use_global_affine=False)
    p.fit(x_source, x_target)
    assert not p.is_global()
    diags = p.diagnostics
    assert len(diags.block_ranges) > 0
    y = p.transform(x_target)
    assert y.shape == x_source.shape
    assert np.allclose(y, x_source, atol=1e-6)
    assert p.estimated_bytes() > 0
    assert p.to_dict()["diagnostics"] is not None


def test_pds_transfer_global_selected():
    x_source, x_target = make_linear_pair(bias=0.0, seed=2)
    p = PDSTransfer(window=4, overlap=0, ridge=1e-12, use_global_affine=True)
    p.fit(x_source, x_target)
    assert p.is_global()
    tc = p.get_global_TC()
    assert tc is not None
    y = p.transform(x_target)
    assert np.allclose(y, x_source, atol=1e-6)
    assert p.estimated_bytes() == tc.nbytes


def test_pds_transfer_global_rejected():
    block = 2
    mat = make_block_diag_matrix(d=6, block=block, seed=5)
    x_source, x_target = make_linear_pair(bias=0.7, seed=3, matrix=mat)
    p = PDSTransfer(window=block, overlap=0, ridge=1e-12, use_global_affine=True)
    p.fit(x_source, x_target)
    assert not p.is_global()
    assert p.to_dict()["diagnostics"] is not None


def test_pds_transfer_clipping_runs():
    mat = make_block_diag_matrix(d=6, block=3, seed=6)
    x_source, x_target = make_linear_pair(bias=0.2, seed=4, matrix=mat)
    p = PDSTransfer(window=3, overlap=0, ridge=1e-12, use_global_affine=False)
    p.fit(x_source, x_target, clip_feature=1.0, clip_target=1.0)
    assert p.to_dict()["diagnostics"] is not None


def test_pds_transfer_setters():
    p = PDSTransfer()
    tc = np.zeros((3, 2))
    p.set_global_TC(tc)
    assert p.is_global()
    assert p.get_global_TC().shape == (3, 2)
    assert p.estimated_bytes() == tc.nbytes
    blocks = [(0, 1, np.zeros((2, 1)))]
    p.set_blocks(blocks)
    assert not p.is_global()
    assert p.get_blocks() == blocks

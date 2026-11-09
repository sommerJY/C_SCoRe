"""Internal shared helpers (rank / correlation / saving)."""
import numpy as np
from scipy.stats import rankdata


def _rank_cols(a):
    return np.apply_along_axis(rankdata, 0, a)


def _corr_cols(A, B):
    """A (n,p), B (n,q) -> (p,q) pearson. constant col -> 0"""
    n = A.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        za = (A - A.mean(0)) / A.std(0)
        zb = (B - B.mean(0)) / B.std(0)
        c = za.T @ zb / n
    return np.nan_to_num(c, nan=0.0)


def _spearman_cols(A, B):
    return _corr_cols(_rank_cols(A), _rank_cols(B))


def _abs_pair_corrs(vals):
    """vals (n, p) -> upper triangle |spearman| 1D"""
    if vals.shape[1] < 2:
        return np.array([0.0])
    c = _spearman_cols(vals, vals)
    iu = np.triu_indices(c.shape[0], k=1)
    return np.abs(c[iu])


def _spearman_mat(X):
    return np.nan_to_num(np.corrcoef(rankdata(X, axis=0), rowvar=False))


def _upper(a):
    return a[np.triu_indices(a.shape[0], k=1)]


def _rank_matrix(S):
    n = len(S)
    iu = np.triu_indices(n, k=1)
    R = np.zeros((n, n))
    R[iu] = rankdata(S[iu])
    return R + R.T, iu


def _unit(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.where(n == 0, 1, n)


def _save(fig, savepath, suffix):
    if savepath:
        fig.savefig(f'{savepath}_{suffix}.png', dpi=300, bbox_inches='tight')
"""CAP change between conditions (generalization of the PCA shift + t-SNE analysis from the Advanced Science paper).

Fit PCA on the overall CAP -> transform the CAPs of conditions A and B onto the same axes
-> for each gene, compute the following from the A -> B movement vector delta
    norm     : |delta| (how much it moved)
    centroid : projection of delta onto the overall direction of movement (centroid difference)
    cent_sim : cosine between delta and the overall direction of movement
    PCdot    : A score . B score
    gene_sim : cosine between the A score and the B score (how much it stays in place)
"""
import colorsys

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from sklearn.decomposition import PCA


def cap_pca_spectrum(cap, n_show=10):
    """Eigenvalues / explained variance of the PCA on the overall CAP (to decide how many PCs to use)"""
    p = PCA().fit(cap.values)
    k = min(n_show, len(p.explained_variance_))
    return pd.DataFrame(dict(PC=np.arange(1, k + 1),
                             eigenvalue=p.explained_variance_[:k],
                             var_ratio=p.explained_variance_ratio_[:k],
                             cum_ratio=np.cumsum(p.explained_variance_ratio_)[:k]))

def cap_shift(caps, base, target, pca_on='all', n_components=3):
    """caps   : caps dict from the run_cscore result
    base   : label of the reference condition (e.g. 'yoked' or 0)
    target : label of the condition to compare (e.g. 'trained' or 1). delta = target - base
    pca_on : CAP to fit the PCA on (default: overall 'all')"""
    labels = [k for k in caps if k != pca_on]
    for x in (base, target):
        if x not in caps:
            raise ValueError(f"There is no CAP for label '{x}'. Available labels: {labels}")
    if base == target:
        raise ValueError("base and target are the same")
    print(f"[shift] {target} - {base} (PCA fit: {pca_on})")
    genes = list(caps[pca_on].index)
    pca = PCA(n_components=n_components).fit(caps[pca_on].loc[genes, genes].values)
    cols = [f'PC{i + 1}' for i in range(n_components)]
    S = {k: pd.DataFrame(pca.transform(caps[k].loc[genes, genes].values), index=genes, columns=cols)
         for k in (pca_on, base, target)}
    S_base, S_target = S[base].values, S[target].values
    delta = S_target - S_base
    norm = np.linalg.norm(delta, axis=1)
    u = S_target.mean(0) - S_base.mean(0)
    u = u / np.linalg.norm(u)
    contrib = delta @ u
    with np.errstate(invalid='ignore', divide='ignore'):
        cent_sim = contrib / norm
        dot = (S_base * S_target).sum(1)
        gene_sim = dot / (np.linalg.norm(S_base, axis=1) * np.linalg.norm(S_target, axis=1))
    shift = pd.DataFrame(dict(norm=norm, centroid=contrib, PCdot=dot,
                              cent_sim=cent_sim, gene_sim=gene_sim), index=genes)
    return dict(pca=pca, scores=S, shift=shift, var_ratio=pca.explained_variance_ratio_,
                base=base, target=target, pca_on=pca_on)

def _brighten(rgb, lightness=0.4):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, max(l, lightness), s)


def shift_tsne(shift, cols=('norm', 'cent_sim', 'gene_sim'), perplexity=30, random_state=5):
    """2D t-SNE from the shift metrics. Returns: comp1, comp2 + the metric columns used"""
    from sklearn.manifold import TSNE
    X = shift[list(cols)].fillna(0.0)
    perplexity = min(perplexity, len(X) - 1)
    Z = TSNE(n_components=2, perplexity=perplexity, random_state=random_state).fit_transform(X.values)
    out = pd.DataFrame(Z, index=X.index, columns=['comp1', 'comp2'])
    return out.join(shift[list(cols)])



def rgb_colors(scores, min_brightness=0.3, brighten=True, invert=None):
    """PC1-3 scores as RGB colors (as in the paper figure). Returns: gene -> hex Series
    invert: channel(s) to flip. e.g. invert='G' or ['R', 'B'] (the paper flips G)"""
    X = scores.iloc[:, :3]
    rng = (X.max() - X.min()).replace(0, 1)
    rgb = (X - X.min()) / rng
    if invert:
        for ch in ([invert] if isinstance(invert, str) else invert):
            i = 'RGB'.index(ch.upper())
            rgb.iloc[:, i] = 1 - rgb.iloc[:, i]
    rgb = rgb * (1 - min_brightness) + min_brightness
    f = _brighten if brighten else (lambda v: v)
    return pd.Series([to_hex(f(tuple(r))) for r in rgb.values], index=X.index)
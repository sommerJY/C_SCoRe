"""C-SCoRe core: association scores based on xi / Spearman, all-pairs matrices, consensus Louvain (CAP).
Does not depend on project paths or metadata. Saves only when out_path is given.
"""
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
from joblib import Parallel, delayed
from tqdm import tqdm


# =====================================================================
# 1. Single pair (for small repeated use, e.g. gene vs target)
# =====================================================================

def xi_cor(x, y, rng=None):
    """Chatterjee xi(x -> y). r_i=#{Y_j<=Y_i}(max), l_i=#{Y_j>=Y_i}(max on -y).
    If X has ties, it is recommended to pass rng (np.random.Generator) to break them randomly."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if rng is not None:
        order = np.lexsort((rng.random(n), x))  # multi-key sort, later keys take priority
    else:
        order = np.argsort(x, kind='stable')
    ys = y[order]
    r = stats.rankdata(ys, method='max')
    l = stats.rankdata(-ys, method='max')
    A = np.abs(np.diff(r)).sum()
    D = 2 * np.sum(l * (n - l))
    if D == 0:
        return 0.0
    return float(np.clip(1 - n * A / D, 0.0, 1.0))


def get_new_xi(n, old_xi):
    """xi_new = xi * (n+1)/(n-2) (exact for perfectly monotone data with no ties)"""
    if n <= 2:
        return old_xi
    new = old_xi * (n + 1) / (n - 2)
    return float(np.clip(new, 0, 1))


def pval_zeta_rayleigh(xi, rho, n):
    z_xi = xi / np.sqrt(0.4 / n)
    z_rho = rho * np.sqrt(n - 1)
    zeta_stat = np.sqrt(z_xi ** 2 + z_rho ** 2)
    p_value = np.exp(-zeta_stat ** 2 / 2)
    return zeta_stat, p_value


def compute_metrics(X_re, Y_re, n, symmetric=False, rng=None):
    X_re = np.asarray(X_re, dtype=float)
    Y_re = np.asarray(Y_re, dtype=float)
    if np.ptp(X_re) == 0 or np.ptp(Y_re) == 0:
        return {'SCOR2': 0, 'PCOR2': 0, 'XI_new': 0, 'Z_pv': 1.0, 'SIGMA': 0}
    pc, pp = stats.pearsonr(X_re, Y_re)
    sc, sp = stats.spearmanr(X_re, Y_re)
    pc2, sc2 = abs(pc), abs(sc)
    if symmetric:
        ox = max(xi_cor(X_re, Y_re, rng), xi_cor(Y_re, X_re, rng))
    else:
        ox = xi_cor(X_re, Y_re, rng)
    nx = get_new_xi(n, ox)
    z_xi = max(0, ox) / np.sqrt(0.4 / n)
    _, zp = pval_zeta_rayleigh(ox, sc2, n)
    return {
        'PCOR': pc, 'P_pv': pp, 'SCOR': sc, 'S_pv': sp,
        'XI_ori': ox, 'XI_new': nx, 'XI_pv': 1 - stats.norm.cdf(z_xi),
        'SIGMA': np.sqrt(sc2 ** 2 + nx ** 2), 'Z_pv': zp,
        'PCOR2': pc2, 'SCOR2': sc2
    }


def gene_vs_target(expr, target, genes=None, n_jobs=10):
    """expr: samples x genes DataFrame, target: values in the same sample order (e.g. behavior PC1).
    Runs compute_metrics(gene, target) for each gene. Returns: DataFrame with an ID column (not saved)."""
    genes = list(expr.columns) if genes is None else list(genes)
    t = np.asarray(target, dtype=float)
    n = len(expr)

    def one(g):
        res = compute_metrics(expr[g], t, n, symmetric=False)
        res['ID'] = g
        return res

    results = Parallel(n_jobs=n_jobs)(delayed(one)(g) for g in tqdm(genes))
    return pd.DataFrame(results)


def select_genes(*tables, p_col='Z_pv', thr=0.05):
    """Union of genes with p<thr across multiple gene_vs_target results."""
    out = []
    for df in tables:
        out += list(df.loc[df[p_col] < thr, 'ID'])
    return list(np.unique(out))


# =====================================================================
# 2. Vectorized all-pairs computation
# =====================================================================

def build_rank_cache(X):
    """X: (p, n) array. Rows = genes, columns = samples."""
    X = np.asarray(X, dtype=float)
    p, n = X.shape
    R_max = stats.rankdata(X, method='max', axis=1)     # numerator r_i = #{Y_j<=Y_i}
    L = stats.rankdata(-X, method='max', axis=1)         # denominator l_i = #{Y_j>=Y_i}
    R_avg = stats.rankdata(X, method='average', axis=1)  # for Spearman
    # Sort with X ties broken randomly (fixed seed for reproducibility)
    rng = np.random.default_rng(0)
    jitter = rng.random(X.shape)
    perm = np.lexsort((jitter, X), axis=1) if False else np.argsort(
        X + jitter * 1e-9, axis=1, kind='stable')
    D = 2.0 * np.sum(L * (n - L), axis=1)
    return dict(R_max=R_max, R_avg=R_avg, perm=perm, D=D, n=n, p=p)


def xi_matrix(cache_x, cache_y=None, n_jobs=1):
    """Asymmetric xi matrix: XI[i, j] = xi(x_i -> y_j)."""
    cy = cache_x if cache_y is None else cache_y
    n = cache_x['n']
    assert n == cy['n'], "Sample counts differ (they must share the same sample axis)"
    R_y, D_y = cy['R_max'], cy['D']
    perm_x = cache_x['perm']
    p_x = cache_x['p']
    valid_D = D_y > 0

    def one_row(i):
        B = R_y[:, perm_x[i]]
        A = np.abs(np.diff(B, axis=1)).sum(axis=1)
        row = np.zeros(cy['p'])
        row[valid_D] = 1.0 - n * A[valid_D] / D_y[valid_D]
        return row

    if n_jobs == 1:
        XI = np.vstack([one_row(i) for i in range(p_x)])
    else:
        rows = Parallel(n_jobs=n_jobs)(delayed(one_row)(i) for i in range(p_x))
        XI = np.vstack(rows)
    return np.clip(XI, 0.0, 1.0)


def full_metric_matrices(X, n_jobs=1):
    """All pairs within one dataset. X: (p, n)."""
    cache = build_rank_cache(X)
    n = cache['n']
    XI_asym = xi_matrix(cache, n_jobs=n_jobs)
    XI_ori = np.maximum(XI_asym, XI_asym.T)
    XI_new = np.clip(XI_ori * (n + 1) / (n - 2), 0.0, 1.0) if n > 2 else XI_ori
    with np.errstate(invalid='ignore'):
        S = np.corrcoef(cache['R_avg'])
        P = np.corrcoef(np.asarray(X, dtype=float))
    S = np.nan_to_num(S)
    P = np.nan_to_num(P)
    SCOR2, PCOR2 = np.abs(S), np.abs(P)
    SIGMA = np.sqrt(SCOR2 ** 2 + XI_new ** 2)
    z_xi = XI_ori / np.sqrt(0.4 / n)
    z_rho = SCOR2 * np.sqrt(n - 1)
    Z_pv = np.exp(-(z_xi ** 2 + z_rho ** 2) / 2.0)
    z_xi_pos = np.maximum(0, XI_ori) / np.sqrt(0.4 / n)
    XI_pv = 1 - stats.norm.cdf(z_xi_pos)
    return dict(XI_ori=XI_ori, XI_new=XI_new, SCOR=S, PCOR=P,
                SCOR2=SCOR2, PCOR2=PCOR2, SIGMA=SIGMA, Z_pv=Z_pv, XI_pv=XI_pv)


def cross_metric_matrices(X_a, X_b, n_jobs=1):
    """(p_a x p_b) matrices between two datasets A and B (same sample axis). symmetric xi = max(xi(a->b), xi(b->a))."""
    cr, cp = build_rank_cache(X_a), build_rank_cache(X_b)
    n = cr['n']
    XI_rp = xi_matrix(cr, cp, n_jobs=n_jobs)
    XI_pr = xi_matrix(cp, cr, n_jobs=n_jobs)
    XI_ori = np.maximum(XI_rp, XI_pr.T)
    XI_new = np.clip(XI_ori * (n + 1) / (n - 2), 0.0, 1.0) if n > 2 else XI_ori
    Rr, Rp = cr['R_avg'], cp['R_avg']

    def _cross_corr(A, B):
        A = A - A.mean(1, keepdims=True)
        B = B - B.mean(1, keepdims=True)
        sa = np.sqrt((A ** 2).sum(1))
        sb = np.sqrt((B ** 2).sum(1))
        with np.errstate(invalid='ignore', divide='ignore'):
            C = (A @ B.T) / np.outer(sa, sb)
        return np.nan_to_num(C)

    S = _cross_corr(Rr, Rp)
    P = _cross_corr(np.asarray(X_a, float), np.asarray(X_b, float))
    SCOR2, PCOR2 = np.abs(S), np.abs(P)
    SIGMA = np.sqrt(SCOR2 ** 2 + XI_new ** 2)
    z_xi = XI_ori / np.sqrt(0.4 / n)
    z_rho = SCOR2 * np.sqrt(n - 1)
    Z_pv = np.exp(-(z_xi ** 2 + z_rho ** 2) / 2.0)
    return dict(XI_ori=XI_ori, XI_new=XI_new, SCOR=S, PCOR=P,
                SCOR2=SCOR2, PCOR2=PCOR2, SIGMA=SIGMA, Z_pv=Z_pv)


def matrices_to_long(mats, genes_a, genes_b=None):
    """Matrices -> long format compatible with the existing CSVs (geneA, geneB, ...)."""
    if genes_b is None:
        iu = np.triu_indices(len(genes_a), k=1)
        rows = {k: v[iu] for k, v in mats.items()}
        df = pd.DataFrame(rows)
        df['geneA'] = np.asarray(genes_a)[iu[0]]
        df['geneB'] = np.asarray(genes_a)[iu[1]]
    else:
        ia, ib = np.meshgrid(np.arange(len(genes_a)), np.arange(len(genes_b)), indexing='ij')
        rows = {k: v.ravel() for k, v in mats.items()}
        df = pd.DataFrame(rows)
        df['geneA'] = np.asarray(genes_a)[ia.ravel()]
        df['geneB'] = np.asarray(genes_b)[ib.ravel()]
    df['MAX'] = np.maximum(df['XI_new'], df['SCOR2'])
    return df


def pairwise_single(expr, genes, n_jobs=8):
    """expr: samples x genes DataFrame -> all-pairs long DataFrame within one omic."""
    genes = list(genes)
    X = expr[genes].astype(float).values.T  # (p, n)
    return matrices_to_long(full_metric_matrices(X, n_jobs=n_jobs), genes)


def pairwise_cross(expr_a, expr_b, genes_a, genes_b, samples=None, n_jobs=8):
    """All-pairs long DataFrame between two omics (same sample axis)."""
    samples = list(expr_a.index) if samples is None else list(samples)
    genes_a, genes_b = list(genes_a), list(genes_b)
    Xa = expr_a.loc[samples, genes_a].astype(float).values.T
    Xb = expr_b.loc[samples, genes_b].astype(float).values.T
    return matrices_to_long(cross_metric_matrices(Xa, Xb, n_jobs=n_jobs), genes_a, genes_b)


# =====================================================================
# 3. Consensus clustering (CAP)
# =====================================================================

def _one_louvain(seed, G, node_index, n):
    import community as community_louvain
    part = community_louvain.best_partition(G, random_state=seed)
    labels = np.empty(n, dtype=np.int32)
    for node, cid in part.items():
        labels[node_index[node]] = cid
    return labels


def graph_process_ckpt(data, out_path=None, n_iter=1000, n_jobs=8,
                       checkpoints=(100, 200, 300, 500),
                       weight_col='SIGMA', edge_percentile=None):
    """Long-format edge table -> Louvain n_iter times -> CAP (gene x gene co-assignment probability).
    - Reports correlation / max|diff| between the cumulative CAP at each checkpoint and the final CAP
    - Reports even/odd seed split-half CAP correlation + Spearman-Brown reliability
    - edge_percentile: e.g. 50 removes the lowest 50% of edges by weight before clustering (default None)
    If out_path is given, saves the CAP csv and a .convergence.txt log."""
    import networkx as nx
    df = data.rename(columns={weight_col: 'weight'}) if weight_col != 'weight' else data
    if edge_percentile is not None:
        thr = np.percentile(df['weight'], edge_percentile)
        df = df[df['weight'] >= thr]
        print(f"[sparsify] {weight_col} >= {thr:.3f} (top {100-edge_percentile}%), edges: {len(df)}")
    G = nx.from_pandas_edgelist(df, 'geneA', 'geneB', edge_attr='weight')
    nodes = list(G.nodes())
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    desc = os.path.basename(out_path) if out_path else 'Clustering'
    all_labels = Parallel(n_jobs=n_jobs)(
        delayed(_one_louvain)(s, G, node_index, n)
        for s in tqdm(range(n_iter), desc=f"Clustering {desc}")
    )
    agg = np.zeros((n, n), dtype=np.uint32)
    agg_even = np.zeros((n, n), dtype=np.uint32)
    agg_odd = np.zeros((n, n), dtype=np.uint32)
    cluster_counts = []
    snapshots = {}
    for k, labels in enumerate(all_labels, start=1):
        local = np.zeros((n, n), dtype=np.uint8)
        for cid in np.unique(labels):
            idx = np.where(labels == cid)[0]
            local[np.ix_(idx, idx)] = 1
        np.fill_diagonal(local, 0)
        agg += local
        if k % 2 == 0:
            agg_even += local
        else:
            agg_odd += local
        cluster_counts.append(len(np.unique(labels)))
        if k in checkpoints:
            snapshots[k] = agg / k
    CAP = agg / n_iter
    iu = np.triu_indices(n, k=1)
    log_lines = [f"[Result] {out_path}"]
    for k in sorted(snapshots):
        r = np.corrcoef(snapshots[k][iu], CAP[iu])[0, 1]
        mad = np.max(np.abs(snapshots[k][iu] - CAP[iu]))
        log_lines.append(f"  corr(CAP_{k}, CAP_{n_iter}) = {r:.5f}, max|diff| = {mad:.4f}")
    half_r = np.corrcoef((agg_even / (n_iter // 2))[iu],
                         (agg_odd / (n_iter - n_iter // 2))[iu])[0, 1]
    sb = 2 * half_r / (1 + half_r)  # Spearman-Brown: reliability estimate for the full number of iterations
    log_lines.append(f"  split-half corr ({n_iter//2} vs {n_iter - n_iter//2}) = {half_r:.5f}"
                     f"  -> full reliability (Spearman-Brown) = {sb:.5f}")
    uniq = sorted(set(cluster_counts))
    log_lines.append(f"  cluster count distribution: {[(s, cluster_counts.count(s)) for s in uniq]}")
    log_text = "\n".join(log_lines)
    print("\n" + log_text)
    co_matrix_df = pd.DataFrame(CAP, index=nodes, columns=nodes)
    if out_path:
        convergence_path = out_path.rsplit('.csv', 1)[0] + '.convergence.txt'
        with open(convergence_path, 'a') as f:
            f.write(log_text + "\n\n")
        co_matrix_df.to_csv(out_path)
    return co_matrix_df
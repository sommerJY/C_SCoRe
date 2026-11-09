"""Module comparison: eigen correlation, module pairing between two datasets (A, B), within-group tests, partition preservation, design comparison, category scores."""
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import rankdata, hypergeom, spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ._utils import _spearman_cols, _abs_pair_corrs, _rank_matrix, _spearman_mat, _unit
from .modules import module_eigengenes, module_symbols


# ----------------------------------------------------------------------
# eigen <-> other omic / behavior, module <-> module
# ----------------------------------------------------------------------
def compute_eigen_correlations(eigen_df, other_df, b):
    """eigengene (module) vs all features of the other omic, vs behavior (spearman)"""
    e = eigen_df.values
    o = other_df.values
    bv = b.values if hasattr(b, 'values') else np.asarray(b)
    eigen_other = _spearman_cols(e, o)
    eigen_beh = _spearman_cols(e, bv.reshape(-1, 1))[:, 0]
    return dict(eigen_other=eigen_other, eigen_beh=eigen_beh, module_ids=list(eigen_df.columns))


def compute_module_module_correlations(eig_a, eig_b):
    """A eigengene x B eigengene Spearman (same samples)"""
    e = eig_a.values
    p = eig_b.values
    corr = _spearman_cols(e, p)
    pval = np.zeros(corr.shape)
    for i in range(e.shape[1]):
        for j in range(p.shape[1]):
            pval[i, j] = stats.spearmanr(e[:, i], p[:, j])[1]
    return dict(module_corr=corr, pval_matrix=pval,
                module_ids_a=list(eig_a.columns),
                module_ids_b=list(eig_b.columns))


def compute_all_module_pair_overlaps(part_a, part_b, mm, id2sym_a=None, id2sym_b=None,
                                     background_genes=None):
    """A module x B module: eigen correlation + gene overlap (hypergeom). If id2sym is None, IDs are compared as they are"""
    id2sym_a, id2sym_b = id2sym_a or {}, id2sym_b or {}
    A_ids, B_ids = mm['module_ids_a'], mm['module_ids_b']
    corr, pcorr = mm['module_corr'], mm['pval_matrix']
    gsets, psets = {}, {}
    for gn, m in part_a.items():
        gsets.setdefault(m, set()).add(id2sym_a.get(gn, gn))
    for pn, m in part_b.items():
        psets.setdefault(m, set()).add(id2sym_b.get(pn, pn))
    if background_genes is None:
        background_genes = set().union(*gsets.values()) | set().union(*psets.values())
    M = len(background_genes)
    rows = []
    for i, gid in enumerate(A_ids):
        A = gsets.get(int(gid.split('_')[1]), set()) & background_genes
        for j, pid in enumerate(B_ids):
            B = psets.get(int(pid.split('_')[1]), set()) & background_genes
            ov = A & B
            n, k, x = len(A), len(B), len(ov)
            if n == 0 or k == 0:
                pv, jac = np.nan, 0.0
            else:
                pv = hypergeom.sf(x - 1, M, n, k)
                jac = x / len(A | B)
            rows.append(dict(module_a=gid, module_b=pid, eigen_corr=corr[i, j],
                             eigen_corr_pval=pcorr[i, j], n_a=n, n_b=k,
                             n_overlap=x, jaccard=jac, overlap_pval=pv, overlap_genes=sorted(ov)))
    return pd.DataFrame(rows)


def get_robust_pairs(df, corr_pval_thresh=0.05, overlap_pval_thresh=0.05, min_n=5):
    f = df[(df['eigen_corr_pval'] < corr_pval_thresh) & (df['overlap_pval'] < overlap_pval_thresh) &
           (df['n_a'] >= min_n) & (df['n_b'] >= min_n)].copy()
    return f.sort_values('eigen_corr', key=abs, ascending=False)


# ----------------------------------------------------------------------
# within-group test (removing the condition effect)
# ----------------------------------------------------------------------
def _group_rank_z(M, groups):
    """Rank within each group -> z. Concatenating and multiplying gives the sample-size-weighted average of within-group Spearman"""
    out = np.zeros(M.shape, dtype=float)
    for g in np.unique(groups):
        m = groups == g
        r = np.apply_along_axis(rankdata, 0, M[m])
        sd = r.std(0)
        sd[sd == 0] = 1
        out[m] = (r - r.mean(0)) / sd
    return out


def within_group_pair_test(eg_r, eg_p, groups, n_perm=5000, seed=0):
    """A eigengene x B eigengene.
    raw: Spearman ignoring groups / obs: pooled within-group Spearman / per_group: per group
    p_cell: per-cell permutation p (B samples are shuffled only within groups)
    p_maxT: multiple-comparison-corrected p, compared against the null of max|r| over all cells"""
    idx = eg_r.index.intersection(eg_p.index)
    R, P = eg_r.loc[idx].values, eg_p.loc[idx].values
    g = np.asarray(groups.loc[idx].values)
    N = len(idx)
    Zr, Zp = _group_rank_z(R, g), _group_rank_z(P, g)
    obs = Zr.T @ Zp / N
    Zr0, Zp0 = _group_rank_z(R, np.zeros(N)), _group_rank_z(P, np.zeros(N))
    raw = Zr0.T @ Zp0 / N
    rng = np.random.default_rng(seed)
    grp_idx = [np.where(g == k)[0] for k in np.unique(g)]
    cnt = np.zeros(obs.shape)
    max_null = np.empty(n_perm)
    for b in range(n_perm):
        perm = np.arange(N)
        for ii in grp_idx:
            perm[ii] = rng.permutation(ii)
        null = Zr.T @ Zp[perm] / N
        cnt += (np.abs(null) >= np.abs(obs))
        max_null[b] = np.abs(null).max()
    p_cell = (cnt + 1) / (n_perm + 1)
    p_maxT = (1 + (max_null[None, None, :] >= np.abs(obs)[:, :, None]).sum(-1)) / (n_perm + 1)
    ri, ci = eg_r.columns, eg_p.columns
    per_group = {k: pd.DataFrame(Zr[g == k].T @ Zp[g == k] / (g == k).sum(), index=ri, columns=ci)
                 for k in np.unique(g)}
    mk = lambda a: pd.DataFrame(a, index=ri, columns=ci)
    return dict(raw=mk(raw), obs=mk(obs), p_cell=mk(p_cell), p_maxT=mk(p_maxT),
                per_group=per_group, n=N, group_n=pd.Series(g).value_counts().to_dict())


def tidy_pairs(res):
    rows = []
    for a in res['obs'].index:
        for b in res['obs'].columns:
            row = dict(module_a=a, module_b=b, raw_r=res['raw'].loc[a, b], within_r=res['obs'].loc[a, b],
                       p_cell=res['p_cell'].loc[a, b], p_maxT=res['p_maxT'].loc[a, b])
            for k, df in res['per_group'].items():
                row[f'grp_{k}'] = df.loc[a, b]
            rows.append(row)
    df = pd.DataFrame(rows)
    gcols = [c for c in df.columns if c.startswith('grp_')]
    df['same_sign_all_groups'] = [bool(np.all(np.sign(r[gcols].values.astype(float)) == np.sign(r['within_r'])))
                                  for _, r in df.iterrows()]
    df['abs_within_r'] = df['within_r'].abs()
    return df.sort_values('p_maxT')


def pair_table_by_groups(expr_a, part_a, expr_b, part_b, groups, id2sym_a=None, id2sym_b=None, n_perm=5000):
    """One table of A module x B module: within-group correlation + overlap within shared genes.
    expr_a, expr_b: samples x features (same samples), part_a, part_b: {feature: module}
    groups: Series indexed by sample (group labels). If id2sym is None, IDs are compared as they are."""
    id2sym_a, id2sym_b = id2sym_a or {}, id2sym_b or {}
    eg_r, qc_r = module_eigengenes(expr_a, part_a)
    eg_p, qc_p = module_eigengenes(expr_b, part_b)
    wg = within_group_pair_test(eg_r, eg_p, groups, n_perm=n_perm)
    tb = tidy_pairs(wg)
    sr = {m: {s.lower() for s in v} for m, v in module_symbols(part_a, id2sym_a).items()}
    sp = {m: {s.lower() for s in v} for m, v in module_symbols(part_b, id2sym_b).items()}
    shared = set().union(*sr.values()) & set().union(*sp.values())
    M = len(shared)
    rows = []
    for a, A in sr.items():
        for b, B in sp.items():
            As, Bs = A & shared, B & shared
            ov = As & Bs
            x, n, k = len(ov), len(As), len(Bs)
            pv = hypergeom.sf(x - 1, M, n, k) if (M and n and k) else np.nan
            rows.append(dict(module_a=f'module_{a}', module_b=f'module_{b}', n_a=len(A), n_b=len(B),
                             n_overlap=x, overlap_ratio=x / min(n, k) if min(n, k) > 0 else 0.0, overlap_p=pv))
    tab = pd.DataFrame(rows).merge(tb, on=['module_a', 'module_b'], how='inner')
    tab['robust'] = (tab['p_maxT'] < 0.05)
    return tab, M, qc_r, qc_p, wg


# ----------------------------------------------------------------------
# partition comparison / preservation
# ----------------------------------------------------------------------
def compare_partitions(partition_a, partition_b, name=''):
    common = sorted(set(partition_a) & set(partition_b))
    la = [partition_a[g] for g in common]
    lb = [partition_b[g] for g in common]
    ari = adjusted_rand_score(la, lb)
    nmi = normalized_mutual_info_score(la, lb)
    print(f"[{name}] n_genes={len(common)}, ARI={ari:.3f}, NMI={nmi:.3f}")
    return dict(ari=ari, nmi=nmi, n_genes=len(common))


def ari_between(pa, pb):
    common = sorted(set(pa) & set(pb))
    la, lb = [pa[g] for g in common], [pb[g] for g in common]
    return adjusted_rand_score(la, lb), normalized_mutual_info_score(la, lb), len(common)


def compute_module_preservation(partition, ref_expr, test_expr, min_genes=3):
    """Mean |spearman| within each module (ref vs test)."""
    cols = list(ref_expr.columns)
    pos = {c: i for i, c in enumerate(cols)}
    rv = ref_expr.values
    tv = test_expr[cols].values
    rows = []
    for m in sorted(set(partition.values())):
        idx = [pos[g] for g, mm in partition.items() if mm == m and g in pos]
        if len(idx) < min_genes:
            continue
        r = _abs_pair_corrs(rv[:, idx])
        t = _abs_pair_corrs(tv[:, idx])
        rows.append(dict(module=f'module_{m}', n_genes=len(idx), ref_coherence=r.mean(),
                         test_coherence=t.mean(), coherence_diff=t.mean() - r.mean()))
    return pd.DataFrame(rows)


def compute_pairwise_preservation(partition_ref, partition_test, gene_list=None):
    """Fraction of gene pairs in the same ref module that are also in the same module in test."""
    common = set(partition_ref) & set(partition_test)
    if gene_list is not None:
        common &= set(gene_list)
    rows = []
    for m in sorted(set(partition_ref[g] for g in common)):
        genes = [g for g in common if partition_ref[g] == m]
        n = len(genes)
        if n < 2:
            rows.append(dict(module=f'module_{m}', n_genes=n, total_pairs=0, preserved_pairs=0, preserve_rate=np.nan))
            continue
        labs = np.array([partition_test[g] for g in genes])
        total = n * (n - 1) // 2
        counts = pd.Series(labs).value_counts().values
        preserved = int(sum(c * (c - 1) // 2 for c in counts))
        rows.append(dict(module=f'module_{m}', n_genes=n, total_pairs=total, preserved_pairs=preserved,
                         preserve_rate=round(preserved / total, 2)))
    return pd.DataFrame(rows)


def raw_vs_module(expr, partition, n_perm=20, seed=0):
    """In the Spearman structure of raw expression, is |r| within modules larger than |r| outside modules?
    ratio = within/between, null_ratio = value when module labels are shuffled (~1)"""
    genes = [g for g in partition if g in expr.columns]
    R = np.abs(_spearman_mat(expr[genes].astype(float).values))
    lab = np.array([partition[g] for g in genes])
    iu = np.triu_indices(len(lab), 1)
    v = R[iu]

    def ratio(l):
        s = (l[:, None] == l[None, :])[iu]
        return v[s].mean() / v[~s].mean(), v[s].mean(), v[~s].mean()

    r, w, b = ratio(lab)
    rng = np.random.default_rng(seed)
    null = np.mean([ratio(rng.permutation(lab))[0] for _ in range(n_perm)])
    return dict(n_genes=len(genes), within=w, between=b, ratio=r, null_ratio=null)


# ----------------------------------------------------------------------
# Design comparison (independent of k)
# ----------------------------------------------------------------------
def mantel(A, B, n_perm=500, seed=0):
    RA, iu = _rank_matrix(np.asarray(A, float))
    RB, _ = _rank_matrix(np.asarray(B, float))
    a, b = RA[iu], RB[iu]
    za = (a - a.mean()) / a.std()
    r = float((za * (b - b.mean()) / b.std()).mean())
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(len(A))
        bp = RB[np.ix_(p, p)][iu]
        null[i] = (za * (bp - bp.mean()) / bp.std()).mean()
    return r, (1 + (null >= r).sum()) / (n_perm + 1)


def cka(A, B):
    def center(S):
        S = np.asarray(S, float)
        return S - S.mean(0, keepdims=True) - S.mean(1, keepdims=True) + S.mean()
    Ac, Bc = center(A), center(B)
    return float((Ac * Bc).sum() / np.sqrt((Ac ** 2).sum() * (Bc ** 2).sum()))


def score_tables(mats, n_perm=500):
    """mats: {label: gene x gene DataFrame} -> Mantel r / p, CKA (label x label)"""
    labs = list(mats)
    idx = mats[labs[0]].index
    for l in labs[1:]:
        idx = idx.intersection(mats[l].index)
    M = {l: mats[l].loc[idx, idx].values.astype(float) for l in labs}
    r = pd.DataFrame(np.eye(len(labs)), index=labs, columns=labs)
    p = pd.DataFrame(0.0, index=labs, columns=labs)
    c = pd.DataFrame(np.eye(len(labs)), index=labs, columns=labs)
    for i, a in enumerate(labs):
        for j in range(i + 1, len(labs)):
            b = labs[j]
            rr, pp = mantel(M[a], M[b], n_perm=n_perm)
            r.loc[a, b] = r.loc[b, a] = rr
            p.loc[a, b] = p.loc[b, a] = pp
            c.loc[a, b] = c.loc[b, a] = cka(M[a], M[b])
    return dict(mantel_r=r, mantel_p=p, cka=c, n_genes=len(idx))


def cap_edge_table(caps):
    """caps: {label: CAP} -> edge Spearman between group CAPs."""
    labs = list(caps)
    idx = caps[labs[0]].index
    for l in labs[1:]:
        idx = idx.intersection(caps[l].index)
    iu = np.triu_indices(len(idx), k=1)
    R = {l: rankdata(caps[l].loc[idx, idx].values[iu]) for l in labs}
    out = pd.DataFrame(np.eye(len(labs)), index=labs, columns=labs)
    for i, a in enumerate(labs):
        for b in labs[i + 1:]:
            out.loc[a, b] = out.loc[b, a] = np.corrcoef(R[a], R[b])[0, 1]
    return out


# ----------------------------------------------------------------------
# Functional profile / category scores
# ----------------------------------------------------------------------
def profile_similarity(mat_r, mat_p, cap=10):
    """Cosine similarity of category enrichment profiles between A modules (mat_r) and B modules (mat_p) (module x module)."""
    cols = mat_r.columns.intersection(mat_p.columns)
    A = np.clip(mat_r[cols].values.astype(float), 0, cap)
    B = np.clip(mat_p[cols].values.astype(float), 0, cap)
    return pd.DataFrame(_unit(A) @ _unit(B).T,
                        index=[f'module_{m}' for m in mat_r.index], columns=[f'module_{m}' for m in mat_p.index])


def global_concordance(link, func_sim, n_perm=5000, seed=0):
    """Spearman between |within_r| and functional profile similarity over all (A module, B module) cells.
    null: shuffle the row/column labels of func_sim independently"""
    link = link.reindex(index=func_sim.index, columns=func_sim.columns)
    a, F = link.values.astype(float), func_sim.values.astype(float)
    if np.nanstd(F) == 0 or np.nanstd(a) == 0:
        return dict(rho=np.nan, p=np.nan, n_cells=a.size)
    rho = spearmanr(a.ravel(), F.ravel())[0]
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        Fp = F[np.ix_(rng.permutation(F.shape[0]), rng.permutation(F.shape[1]))]
        null[i] = spearmanr(a.ravel(), Fp.ravel())[0]
    return dict(rho=float(rho), p=float((1 + (null >= rho).sum()) / (n_perm + 1)), n_cells=a.size)


def category_scores(expr, id2sym, cat_genes, min_size=3):
    """Per-sample category score = mean z-score of the category's genes. expr: samples x genes(ID)"""
    sym = expr.rename(columns=lambda g: str(id2sym.get(g, g)).lower())
    sym = sym.T.groupby(level=0).mean().T
    Z = (sym - sym.mean()) / sym.std(ddof=0).replace(0, np.nan)
    out = {}
    for c, gs in cat_genes.items():
        present = [g for g in gs if g in Z.columns]
        if len(present) >= min_size:
            out[c] = Z[present].mean(axis=1)
    return pd.DataFrame(out)


def category_diag_test(sr, sp, groups, n_perm=2000, seed=0):
    """A category scores (sr) vs B category scores (sp) (within-group Spearman).
    Statistic = mean(diag) - mean(offdiag), null = B samples shuffled within groups (one-sided)."""
    cats = [c for c in sr.columns if c in sp.columns]
    idx = sr.index.intersection(sp.index)
    if len(cats) < 2 or len(idx) < 6:
        return dict(delta=np.nan, p=np.nan, table=pd.DataFrame(), C=pd.DataFrame())
    R, P = sr.loc[idx, cats].values, sp.loc[idx, cats].values
    g = np.asarray(groups.loc[idx].values)
    N = len(idx)
    Zr, Zp = _group_rank_z(R, g), _group_rank_z(P, g)

    def stat(Zp_):
        C = Zr.T @ Zp_ / N
        d = np.diag(C)
        off = (C.sum() - d.sum()) / (C.size - len(d))
        return d.mean() - off, d

    delta, d = stat(Zp)
    C = Zr.T @ Zp / N
    rng = np.random.default_rng(seed)
    grp_idx = [np.where(g == k)[0] for k in np.unique(g)]
    nd, ndiag = np.empty(n_perm), np.empty((n_perm, len(cats)))
    for b in range(n_perm):
        perm = np.arange(N)
        for ii in grp_idx:
            perm[ii] = rng.permutation(ii)
        nd[b], ndiag[b] = stat(Zp[perm])
    p_cat = (1 + (ndiag >= d[None, :]).sum(0)) / (n_perm + 1)
    beaten = [(C[i, i] > np.delete(C[i], i)).mean() for i in range(len(cats))]
    tab = pd.DataFrame(dict(category=cats, r_diag=d, p_perm=p_cat, frac_beaten=beaten))
    tab['fdr'] = multipletests(tab['p_perm'], method='fdr_bh')[1]
    return dict(delta=float(delta), p=float((1 + (nd >= delta).sum()) / (n_perm + 1)),
                table=tab.sort_values('p_perm'), C=pd.DataFrame(C, index=cats, columns=cats))
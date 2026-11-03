"""모듈 비교: eigen 상관, RNA-PROT 모듈 짝, within-group 검정, partition 보존, 설계 비교, category 점수."""
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
    """eigengene (module) vs 다른 omic 전체 feature, vs behavior (spearman)"""
    e = eigen_df.values
    o = other_df.values
    bv = b.values if hasattr(b, 'values') else np.asarray(b)
    eigen_other = _spearman_cols(e, o)
    eigen_beh = _spearman_cols(e, bv.reshape(-1, 1))[:, 0]
    return dict(eigen_other=eigen_other, eigen_beh=eigen_beh, module_ids=list(eigen_df.columns))


def compute_module_module_correlations(eigengene_df, eigenprot_df):
    e = eigengene_df.values
    p = eigenprot_df.values
    corr = _spearman_cols(e, p)
    pval = np.zeros(corr.shape)
    for i in range(e.shape[1]):
        for j in range(p.shape[1]):
            pval[i, j] = stats.spearmanr(e[:, i], p[:, j])[1]
    return dict(module_corr=corr, pval_matrix=pval,
                gene_module_ids=list(eigengene_df.columns),
                protein_module_ids=list(eigenprot_df.columns))


def compute_all_module_pair_overlaps(gene_partition, prot_partition, RNA_id2sym, Prot_id2sym, mm,
                                     background_genes=None):
    g_ids, p_ids = mm['gene_module_ids'], mm['protein_module_ids']
    corr, pcorr = mm['module_corr'], mm['pval_matrix']
    gsets, psets = {}, {}
    for gn, m in gene_partition.items():
        gsets.setdefault(m, set()).add(RNA_id2sym.get(gn, gn))
    for pn, m in prot_partition.items():
        psets.setdefault(m, set()).add(Prot_id2sym.get(pn, pn))
    if background_genes is None:
        background_genes = set().union(*gsets.values()) | set().union(*psets.values())
    M = len(background_genes)
    rows = []
    for i, gid in enumerate(g_ids):
        A = gsets.get(int(gid.split('_')[1]), set()) & background_genes
        for j, pid in enumerate(p_ids):
            B = psets.get(int(pid.split('_')[1]), set()) & background_genes
            ov = A & B
            n, k, x = len(A), len(B), len(ov)
            if n == 0 or k == 0:
                pv, jac = np.nan, 0.0
            else:
                pv = hypergeom.sf(x - 1, M, n, k)
                jac = x / len(A | B)
            rows.append(dict(gene_module=gid, prot_module=pid, eigen_corr=corr[i, j],
                             eigen_corr_pval=pcorr[i, j], n_gene_module=n, n_prot_module=k,
                             n_overlap=x, jaccard=jac, overlap_pval=pv, overlap_genes=sorted(ov)))
    return pd.DataFrame(rows)


def get_robust_pairs(df, corr_pval_thresh=0.05, overlap_pval_thresh=0.05, min_n=5):
    f = df[(df['eigen_corr_pval'] < corr_pval_thresh) & (df['overlap_pval'] < overlap_pval_thresh) &
           (df['n_gene_module'] >= min_n) & (df['n_prot_module'] >= min_n)].copy()
    return f.sort_values('eigen_corr', key=abs, ascending=False)


# ----------------------------------------------------------------------
# within-group test (조건 효과 제거)
# ----------------------------------------------------------------------
def _group_rank_z(M, groups):
    """그룹 안에서 rank -> z. 이어붙여 곱하면 '그룹 내 Spearman 의 표본수 가중 평균'"""
    out = np.zeros(M.shape, dtype=float)
    for g in np.unique(groups):
        m = groups == g
        r = np.apply_along_axis(rankdata, 0, M[m])
        sd = r.std(0)
        sd[sd == 0] = 1
        out[m] = (r - r.mean(0)) / sd
    return out


def within_group_pair_test(eg_r, eg_p, groups, n_perm=5000, seed=0):
    """RNA eigengene x PROT eigengene.
    raw: 그룹 무시 Spearman / obs: 그룹 내 pooled Spearman / per_group: 그룹별
    p_cell: 칸별 permutation p (PROT 샘플을 그룹 안에서만 섞음)
    p_maxT: 전체 칸 max|r| null 과 비교한 다중비교 보정 p"""
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
            row = dict(rna_module=a, prot_module=b, raw_r=res['raw'].loc[a, b], within_r=res['obs'].loc[a, b],
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


def pair_table_by_groups(cond_r, cond_p, groups, id2sym_r, id2sym_p, n_perm=5000):
    """RNA 모듈 x PROT 모듈 한 표: within-group 상관 + 공유 유전자 안에서의 overlap.
    groups: 샘플 index 를 가진 Series (그룹 라벨)."""
    eg_r, qc_r = module_eigengenes(cond_r['expr']['both'], cond_r['partition'])
    eg_p, qc_p = module_eigengenes(cond_p['expr']['both'], cond_p['partition'])
    wg = within_group_pair_test(eg_r, eg_p, groups, n_perm=n_perm)
    tb = tidy_pairs(wg)
    sr = {m: {s.lower() for s in v} for m, v in module_symbols(cond_r['partition'], id2sym_r).items()}
    sp = {m: {s.lower() for s in v} for m, v in module_symbols(cond_p['partition'], id2sym_p).items()}
    shared = set().union(*sr.values()) & set().union(*sp.values())
    M = len(shared)
    rows = []
    for a, A in sr.items():
        for b, B in sp.items():
            As, Bs = A & shared, B & shared
            ov = As & Bs
            x, n, k = len(ov), len(As), len(Bs)
            pv = hypergeom.sf(x - 1, M, n, k) if (M and n and k) else np.nan
            rows.append(dict(rna_module=f'module_{a}', prot_module=f'module_{b}', n_rna=len(A), n_prot=len(B),
                             n_overlap=x, overlap_ratio=x / min(n, k) if min(n, k) > 0 else 0.0, overlap_p=pv))
    tab = pd.DataFrame(rows).merge(tb, on=['rna_module', 'prot_module'], how='inner')
    tab['robust'] = (tab['p_maxT'] < 0.05)
    return tab, M, qc_r, qc_p, wg


# ----------------------------------------------------------------------
# partition 비교 / 보존
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
    """모듈 내부 mean |spearman| (ref vs test)."""
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
    """ref 모듈 안 gene pair 가 test 에서도 같은 모듈인 비율."""
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
    """raw expression 의 Spearman 구조에서 모듈 안 |r| 이 모듈 밖 |r| 보다 큰가.
    ratio = within/between, null_ratio = 모듈 라벨을 섞었을 때 (~1)"""
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
# 설계 비교 (k 와 무관)
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
    """caps: {label: CAP} -> 그룹 CAP 끼리 edge Spearman."""
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
# 기능 프로파일 / category 점수
# ----------------------------------------------------------------------
def profile_similarity(mat_r, mat_p, cap=10):
    """RNA 모듈과 PROT 모듈의 category enrichment 프로파일 cosine 유사도 (module x module)."""
    cols = mat_r.columns.intersection(mat_p.columns)
    A = np.clip(mat_r[cols].values.astype(float), 0, cap)
    B = np.clip(mat_p[cols].values.astype(float), 0, cap)
    return pd.DataFrame(_unit(A) @ _unit(B).T,
                        index=[f'module_{m}' for m in mat_r.index], columns=[f'module_{m}' for m in mat_p.index])


def global_concordance(link, func_sim, n_perm=5000, seed=0):
    """모든 (RNA 모듈, PROT 모듈) 칸에서 |within_r| 와 기능 프로파일 유사도의 Spearman.
    null: func_sim 의 행/열 라벨을 각각 독립적으로 섞음"""
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
    """샘플별 category 점수 = category 유전자 z-score 평균. expr: samples x genes(ID)"""
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
    """RNA category 점수 vs PROT category 점수 (그룹 내 Spearman).
    통계 = mean(diag) - mean(offdiag), null = PROT 샘플을 그룹 안에서 섞음 (one-sided)."""
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

"""기능 annotation: gene set enrichment, gProfiler, term clustering."""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests

from .modules import module_symbols

THR = -np.log10(0.05)
SRC = ['GO:BP', 'GO:CC', 'GO:MF', 'KEGG', 'REAC']
EMPTY_COLS = ['module', 'source', 'native', 'name', 'p_value', 'term_size', 'intersection_size']


# ----------------------------------------------------------------------
# gene set enrichment (모듈별로 주어진 유전자 목록이 몰려 있는지)
# ----------------------------------------------------------------------
def gene_set_enrichment(partition, id_dict, target_genes, background_genes=None,
                        n_perm=2000, seed=0, fdr_method='fdr_bh'):
    """모듈별로 target_genes 가 얼마나 몰려 있는지 (hypergeom + permutation)."""
    rng = np.random.default_rng(seed)
    target_lower = set(s.lower() for s in target_genes)
    module_genes = {}
    for name, m in partition.items():
        module_genes.setdefault(m, set()).add(id_dict.get(name, name))
    if background_genes is None:
        background_genes = set().union(*module_genes.values())
    bg_list = sorted(set(s.lower() for s in background_genes))
    is_target = np.array([g in target_lower for g in bg_list])
    M, K = len(bg_list), int(is_target.sum())
    bg_set = set(bg_list)
    expected = K / M if M > 0 else np.nan
    rows = []
    for m, genes in module_genes.items():
        gl = set(g.lower() for g in genes) & bg_set
        n = len(gl)
        l2o = {g.lower(): g for g in genes}
        ov = sorted(l2o[g] for g in gl if g in target_lower)
        x = len(ov)
        if n == 0:
            hp = pp = nm = fe = np.nan
        else:
            hp = hypergeom.sf(x - 1, M, K, n)
            if n_perm:
                null = np.array([is_target[rng.choice(M, size=n, replace=False)].sum() for _ in range(n_perm)])
                pp = (np.sum(null >= x) + 1) / (n_perm + 1)
                nm = null.mean()
            else:
                pp = nm = np.nan
            fe = (x / n) / expected if expected > 0 else np.nan
        rows.append(dict(module=f'module_{m}', n_module=n, n_target_in_module=x,
                         pct_target=(x / n) if n else np.nan, expected_pct_target=expected,
                         fold_enrichment=fe, hyper_pval=hp, perm_pval=pp, perm_null_mean=nm,
                         target_genes_in_module=ov))
    df = pd.DataFrame(rows)
    for col, fcol in [('hyper_pval', 'hyper_fdr'), ('perm_pval', 'perm_fdr')]:
        df[fcol] = np.nan
        mk = df[col].notna()
        if mk.sum() > 0:
            df.loc[mk, fcol] = multipletests(df.loc[mk, col], method=fdr_method)[1]
    return df.sort_values('hyper_pval').reset_index(drop=True)




# ----------------------------------------------------------------------
# gProfiler
# ----------------------------------------------------------------------
def run_gprofiler(genes, organism='mmusculus', min_term=5, max_term=1000, min_inter=3):
    from gprofiler import GProfiler
    res = GProfiler(return_dataframe=True).profile(
        organism=organism, query=list(genes), user_threshold=0.05,
        significance_threshold_method='fdr', no_evidences=False)
    filt = res[(res.term_size < max_term) & (res.term_size > min_term) & (res.intersection_size > min_inter)]
    cols = ['name', 'p_value', 'term_size', 'intersection_size', 'source']
    return res[cols], filt[cols]


def module_terms(gp, partition, id2sym, min_term=10, max_term=500, min_inter=3, background=None,
                 organism='mmusculus'):
    """모듈마다 gProfiler.
    background=None : 그 partition 에 들어간 유전자 전체 (보수적)
    background=list : 그 omic 에서 검출된 전체 유전자 symbol 등 (검정력 높음)"""
    mods = module_symbols(partition, id2sym)
    allg = {s for v in mods.values() for s in v}
    bg = sorted(allg if background is None else (set(background) | allg))
    rows = []
    for m, genes in mods.items():
        if len(genes) < 5:
            continue
        res = gp.profile(organism=organism, query=genes, sources=SRC, user_threshold=0.05,
                         significance_threshold_method='fdr', no_evidences=True,
                         background=bg, domain_scope='custom')
        if res is None or len(res) == 0:
            continue
        res = res[(res.term_size >= min_term) & (res.term_size <= max_term) & (res.intersection_size >= min_inter)]
        if len(res):
            rows.append(res.assign(module=m)[EMPTY_COLS])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=EMPTY_COLS)


def symbol_check(gp, partition, id2sym, organism='mmusculus'):
    """term hit 이 0 일 때 먼저 볼 것: symbol 이 비었거나 ';' 로 묶였거나 ID 그대로인지"""
    raw = {g: id2sym.get(g, g) for g in partition}
    ok = {g: s for g, s in raw.items() if isinstance(s, str) and s != ''}
    semi = [s for s in ok.values() if ';' in s]
    out = dict(n_genes=len(raw), n_missing=len(raw) - len(ok), n_with_semicolon=len(semi),
               n_symbol_equals_id=int(sum(1 for g, s in ok.items() if s == g)),
               n_unique=len(set(ok.values())), example_semicolon=semi[:3], example=list(ok.values())[:3])
    try:
        conv = gp.convert(organism=organism, query=sorted(set(ok.values())), target_namespace='ENSG')
        out['n_mapped_by_gprofiler'] = int((conv['converted'].astype(str) != 'None').sum())
    except Exception as e:
        out['n_mapped_by_gprofiler'] = f'check failed: {type(e).__name__}'
    return out


def term_diagnostics(gp, partition, id2sym, universe, organism='mmusculus'):
    """모듈별로 (a) partition 배경 (b) 검출 전체 유전자 배경 에서 term hit 수 비교."""
    mods = module_symbols(partition, id2sym)
    allg = {s for v in mods.values() for s in v}
    bgs = {'partition': sorted(allg), 'detected': sorted(set(universe) | allg)}
    rows = []
    for m, genes in mods.items():
        if len(genes) < 5:
            continue
        row = dict(module=m, n_genes=len(genes))
        for name, bg in bgs.items():
            res = gp.profile(organism=organism, query=genes, sources=SRC, user_threshold=0.05,
                             significance_threshold_method='fdr', no_evidences=True,
                             background=bg, domain_scope='custom')
            n = 0 if res is None else len(res)
            row[f'{name}_bg_size'] = len(bg)
            row[f'{name}_hits_raw'] = n
            row[f'{name}_hits_filtered'] = int(((res.term_size >= 10) & (res.term_size <= 500) &
                                                (res.intersection_size >= 3)).sum()) if n else 0
            row[f'{name}_min_p'] = float(res['p_value'].min()) if n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# term clustering -> category
# ----------------------------------------------------------------------
def build_term_pool(terms_dict):
    allhits = pd.concat([df.assign(key=[k] * len(df)) for k, df in terms_dict.items() if len(df)],
                        ignore_index=True)
    allhits['score'] = -np.log10(allhits['p_value'].astype(float).clip(lower=1e-300))
    return allhits.groupby('native').agg(name=('name', 'first'), source=('source', 'first'),
                                         n_hits=('module', 'size'), sum_score=('score', 'sum'))


def cluster_terms_nested(pool, term_genes, cuts=(0.3, 0.1)):
    """term 유전자 Jaccard + average linkage 를 한 번 만들고, 두 높이에서 자름 (nested 보장)"""
    ids = [t for t in pool.index if t in term_genes and len(term_genes[t]) >= 3]
    genes = sorted({g for t in ids for g in term_genes[t]})
    gi = {g: i for i, g in enumerate(genes)}
    rows, cols = [], []
    for i, t in enumerate(ids):
        for g in term_genes[t]:
            rows.append(i)
            cols.append(gi[g])
    B = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), len(genes)))
    inter = (B @ B.T).toarray()
    s = np.asarray(B.sum(1)).ravel()
    union = s[:, None] + s[None, :] - inter
    J = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    dist = 1 - J
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), 'average')
    out = pd.DataFrame(index=ids)
    for name, c in zip(('fine', 'coarse'), cuts):
        out[name] = fcluster(Z, t=1 - c, criterion='distance')
    return out


def summarize_clusters(pool, assign, level):
    d = pool.loc[assign.index].assign(cluster=assign[level])
    rows = []
    for c, g in d.groupby('cluster'):
        top = g['sum_score'].idxmax()
        rows.append(dict(cluster=c, n_terms=len(g), n_hits=int(g['n_hits'].sum()), sum_score=g['sum_score'].sum(),
                         rep_term=top, rep_name=g.loc[top, 'name'],
                         top_terms=' | '.join(g.sort_values('sum_score', ascending=False)['name'].head(8))))
    return pd.DataFrame(rows).set_index('cluster').sort_values('n_hits', ascending=False)


def cluster_gene_sets(term2cat, term_genes, cat_label):
    """{label: 그 군집에 속한 모든 term 의 유전자 합집합(lower symbol)}"""
    out = {}
    for cid, lab in cat_label.items():
        ts = term2cat.index[term2cat == cid]
        out[lab] = set().union(*(term_genes[t] for t in ts)) if len(ts) else set()
    return out


def module_category_matrix(hits, term2cat, cat_label, mods):
    h = hits[hits['native'].isin(term2cat.index)].copy()
    if len(h) == 0:
        return pd.DataFrame(0.0, index=mods, columns=list(cat_label.values()))
    h['cat'] = h['native'].map(term2cat)
    h['score'] = -np.log10(h['p_value'].astype(float).clip(lower=1e-300))
    mat = h.pivot_table(index='module', columns='cat', values='score', aggfunc='max')
    mat = mat.reindex(index=mods, columns=list(cat_label)).fillna(0)
    mat.columns = [cat_label[c] for c in mat.columns]
    return mat


def label_modules(mat, thr=THR):
    return mat.idxmax(axis=1).where(mat.max(axis=1) >= thr, 'unannotated')

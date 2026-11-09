"""Run C-SCoRe in one go with a fixed input format.

Input format (all indexes are the same sample IDs)
    expr  : samples x genes DataFrame (numeric only)
    beh   : 1-column DataFrame or Series (e.g. behavior PC1). None if not available
    conds : 1-column DataFrame or Series (e.g. training_class 0/1). None if not available
"""
import os

import numpy as np
import pandas as pd

from .cap import gene_vs_target, select_genes, pairwise_single, graph_process_ckpt


def _as_series(x, name):
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"{name} must have exactly 1 column (currently {x.shape[1]})")
        x = x.iloc[:, 0]
    if not isinstance(x, pd.Series):
        raise ValueError(f"{name} must be a DataFrame (1 column) or a Series")
    return x


def check_inputs(expr, beh=None, conds=None):
    """Check the format + align to the same sample order. Returns: (expr, beh Series, conds Series)"""
    if not isinstance(expr, pd.DataFrame):
        raise ValueError("expr must be a samples x genes DataFrame")
    if expr.index.has_duplicates:
        raise ValueError("expr index (sample IDs) has duplicates")
    if expr.columns.has_duplicates:
        raise ValueError("expr columns (genes) have duplicates")
    non_num = [c for c in expr.columns if not pd.api.types.is_numeric_dtype(expr[c])]
    if non_num:
        raise ValueError(f"expr has non-numeric columns: {non_num[:5]}")
    beh = _as_series(beh, 'beh')
    conds = _as_series(conds, 'conds')
    for name, s in [('beh', beh), ('conds', conds)]:
        if s is None:
            continue
        missing = set(expr.index) - set(s.index)
        if missing:
            raise ValueError(f"{name} is missing some samples: {sorted(missing)[:5]}")
        if s.loc[expr.index].isna().any():
            raise ValueError(f"{name} has NaN")
    beh = None if beh is None else beh.loc[expr.index].astype(float)
    conds = None if conds is None else conds.loc[expr.index]
    const = expr.columns[expr.std() == 0]
    if len(const):
        print(f"[check] excluded {len(const)} genes with identical values in all samples")
        expr = expr.drop(columns=const)
    msg = f"[check] samples={expr.shape[0]}, genes={expr.shape[1]}"
    if conds is not None:
        msg += f", conds={conds.value_counts().sort_index().to_dict()}"
    print(msg)
    return expr, beh, conds


def run_cscore(expr, beh=None, conds=None, genes=None, add_genes=None, p_thr=0.05,
               n_iter=1000, n_jobs=8, out_dir=None, prefix=''):
    """Gene selection -> overall CAP + per-condition sub CAPs.
    genes    : gene list given directly. If None, genes with Z_pv < p_thr against beh (using all samples)
    add_genes: genes to always include (e.g. candidate genes)
    out_dir  : if given, saves gene score / pairwise / CAP csv
    Returns dict: expr, beh, conds, genes, gene_scores, caps={'all': CAP, <cond value>: CAP, ...}"""
    expr, beh, conds = check_inputs(expr, beh, conds)

    gene_scores = None
    if genes is None:
        if beh is None:
            raise ValueError("Neither genes nor beh was given. One of them is required")
        gene_scores = gene_vs_target(expr, beh, n_jobs=n_jobs)
        genes = select_genes(gene_scores, thr=p_thr)
        print(f"[genes] Z_pv < {p_thr}: {len(genes)}")
    genes = [g for g in genes if g in expr.columns]
    if add_genes:
        extra = [g for g in add_genes if g in expr.columns and g not in genes]
        genes = genes + extra
        print(f"[genes] add_genes added: {len(extra)}")
    genes = list(np.unique(genes))
    print(f"[genes] final {len(genes)}")

    def path(name):
        return None if out_dir is None else os.path.join(out_dir, f'{prefix}{name}.csv')

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        if gene_scores is not None:
            gene_scores.to_csv(path('gene_scores'))

    sets = [('all', expr.index)]
    if conds is not None:
        sets += [(c, conds.index[conds == c]) for c in sorted(conds.unique())]

    caps = {}
    for label, idx in sets:
        print(f"\n[CAP] {label}: n={len(idx)}")
        long = pairwise_single(expr.loc[idx], genes, n_jobs=n_jobs)
        if out_dir is not None:
            long.to_csv(path(f'pairwise_{label}'))
        cap = graph_process_ckpt(long, path(f'CAP_{label}'), n_iter=n_iter, n_jobs=n_jobs)
        cap = cap.reindex(index=genes, columns=genes).fillna(0.0)
        a = cap.values.copy()
        np.fill_diagonal(a, 1.0)
        caps[label] = pd.DataFrame(a, index=genes, columns=genes)

    return dict(expr=expr, beh=beh, conds=conds, genes=genes, gene_scores=gene_scores, caps=caps)
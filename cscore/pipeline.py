"""고정 입력 포맷으로 C-SCoRe 한 번에 돌리기.

입력 포맷 (index 는 모두 같은 샘플 ID)
    expr  : samples x genes DataFrame (숫자만)
    beh   : 1 컬럼 DataFrame 또는 Series (예: 행동 PC1). 없으면 None
    conds : 1 컬럼 DataFrame 또는 Series (예: training_class 0/1). 없으면 None
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
            raise ValueError(f"{name} 는 컬럼이 1개여야 해요 (지금 {x.shape[1]}개)")
        x = x.iloc[:, 0]
    if not isinstance(x, pd.Series):
        raise ValueError(f"{name} 는 DataFrame(1 컬럼) 또는 Series 여야 해요")
    return x


def check_inputs(expr, beh=None, conds=None):
    """포맷 확인 + 같은 샘플 순서로 맞추기. 반환: (expr, beh Series, conds Series)"""
    if not isinstance(expr, pd.DataFrame):
        raise ValueError("expr 는 samples x genes DataFrame 이어야 해요")
    if expr.index.has_duplicates:
        raise ValueError("expr index (샘플 ID) 에 중복이 있어요")
    if expr.columns.has_duplicates:
        raise ValueError("expr 컬럼 (유전자) 에 중복이 있어요")
    non_num = [c for c in expr.columns if not pd.api.types.is_numeric_dtype(expr[c])]
    if non_num:
        raise ValueError(f"expr 에 숫자가 아닌 컬럼이 있어요: {non_num[:5]}")
    beh = _as_series(beh, 'beh')
    conds = _as_series(conds, 'conds')
    for name, s in [('beh', beh), ('conds', conds)]:
        if s is None:
            continue
        missing = set(expr.index) - set(s.index)
        if missing:
            raise ValueError(f"{name} 에 없는 샘플이 있어요: {sorted(missing)[:5]}")
        if s.loc[expr.index].isna().any():
            raise ValueError(f"{name} 에 NaN 이 있어요")
    beh = None if beh is None else beh.loc[expr.index].astype(float)
    conds = None if conds is None else conds.loc[expr.index]
    const = expr.columns[expr.std() == 0]
    if len(const):
        print(f"[check] 모든 샘플에서 값이 같은 유전자 {len(const)}개 제외")
        expr = expr.drop(columns=const)
    msg = f"[check] samples={expr.shape[0]}, genes={expr.shape[1]}"
    if conds is not None:
        msg += f", conds={conds.value_counts().sort_index().to_dict()}"
    print(msg)
    return expr, beh, conds


def run_cscore(expr, beh=None, conds=None, genes=None, add_genes=None, p_thr=0.05,
               n_iter=1000, n_jobs=8, out_dir=None, prefix=''):
    """유전자 선정 -> 전체 CAP + 조건별 sub CAP.
    genes    : 직접 줄 유전자 목록. None 이면 beh 와 Z_pv < p_thr 인 유전자 (전체 샘플 기준)
    add_genes: 항상 포함할 유전자 (예: candidate gene)
    out_dir  : 주면 gene score / pairwise / CAP csv 저장
    반환 dict: expr, beh, conds, genes, gene_scores, caps={'all': CAP, <cond값>: CAP, ...}"""
    expr, beh, conds = check_inputs(expr, beh, conds)

    gene_scores = None
    if genes is None:
        if beh is None:
            raise ValueError("genes 도 beh 도 없어요. 둘 중 하나는 필요해요")
        gene_scores = gene_vs_target(expr, beh, n_jobs=n_jobs)
        genes = select_genes(gene_scores, thr=p_thr)
        print(f"[genes] Z_pv < {p_thr}: {len(genes)}")
    genes = [g for g in genes if g in expr.columns]
    if add_genes:
        extra = [g for g in add_genes if g in expr.columns and g not in genes]
        genes = genes + extra
        print(f"[genes] add_genes 추가: {len(extra)}")
    genes = list(np.unique(genes))
    print(f"[genes] 최종 {len(genes)}")

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

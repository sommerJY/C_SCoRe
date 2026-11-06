# cscore

C-SCoRe (Correlation Signal Co-cluster Reduction) 분석 패키지.

## 설치 (처음 한 번)

```bash
cd cscore_pkg
pip install -e .
```

`-e` (editable) 로 설치하면 `cscore/` 안의 코드를 고쳐도 다시 설치할 필요 없이 바로 반영됨.
gProfiler / sunburst 까지 쓰려면 `pip install -e ".[enrich]"`.

## 구조

```
cscore/
  cap.py       xi_cor, pairwise 행렬, consensus Louvain (CAP)
  modules.py   CAP -> 모듈 (spectral, k 선택, eigengene)
  compare.py   모듈 비교, within-group 검정, partition 보존, Mantel/CKA
  enrich.py    gene set enrichment, gProfiler, term clustering
  plotting.py  그림
```

## 사용

```python
import cscore as cs

res  = cs.gene_vs_target(expr, behavior_pc1)          # 유전자 vs 행동
sel  = cs.select_genes(res)                            # Z_pv < 0.05
long = cs.pairwise_single(expr, sel)                   # all-pairs
cap  = cs.graph_process_ckpt(long, 'CAP.csv', n_iter=1000)
```

## 고정 입력 포맷으로 한 번에

```python
# 샘플 ID 가 index, 셋 다 같은 샘플
expr  = df[gene_list];          expr.index  = df.RNAseqID   # samples x genes
beh   = df[['PC1']];            beh.index   = df.RNAseqID   # 1 컬럼
conds = df[['training_class']]; conds.index = df.RNAseqID   # 1 컬럼 (예: 0/1)

out = cs.run_cscore(expr, beh, conds, add_genes=candidates, n_iter=1000, out_dir='results/')
# out['caps'] = {'all': CAP, 0: CAP, 1: CAP}

spec = cs.cap_pca_spectrum(out['caps']['all']);  cs.plot_pc_spectrum(spec)
res  = cs.cap_shift(out['caps'], order=(0, 1), n_components=3)   # 1 - 0
cs.plot_cap_pca_3d(res, highlight=candidates, label_a='Yoked', label_b='Trained')

ts = cs.shift_tsne(res['shift'], perplexity=30, random_state=5)
cs.plot_shift_tsne(ts, color_col='norm', size_col='norm', highlight=candidates)
```

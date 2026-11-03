# cscore

C-SCoRe (Correlation Signal Co-cluster Reduction) 

## install

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


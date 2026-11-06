# cscore

C-SCoRe (Correlation Signal Co-cluster Reduction) 분석 패키지.

## 0. 설치 / 불러오기

### 필요한 패키지

| 용도 | 패키지 (pip 이름) |
|---|---|
| 기본 계산 | `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`, `joblib`, `tqdm` |
| CAP (Louvain) | `networkx`, `python-louvain` (import 이름은 `community`) |
| 그림 | `matplotlib`, `seaborn` |
| MCL, Sankey (7단계) | `markov_clustering`, `plotly` |
| GO 분석 (enrich.py) | `gprofiler-official` |

Python 3.9 이상.

### 방법 1: 패키지로 설치 (권장)

```bash
cd cscore_pkg
pip install -e .              # 기본 패키지까지 같이 설치됨. 코드를 고쳐도 다시 설치할 필요 없음
pip install -e ".[enrich]"    # MCL, Sankey, GO 까지 쓸 때
```

### 방법 2: 설치 없이 폴더째 쓰기

필요한 패키지를 직접 설치한 뒤 경로를 추가해서 불러옴.

```bash
pip install numpy pandas scipy statsmodels scikit-learn joblib tqdm \
            networkx python-louvain matplotlib seaborn \
            markov_clustering plotly gprofiler-official
```

conda 환경을 쓴다면 그 환경을 켠 상태에서 설치할 것 (`conda activate 환경이름`).
설치했는데 `No module named ...` 가 나오면 다른 환경에 설치된 것.
Python 안에서 `import sys; print(sys.executable)` 로 지금 환경을 확인하고,
그 경로로 `/.../bin/python -m pip install 패키지` 처럼 설치하면 확실함.

주의: `pip install community` 는 다른 패키지라서 Louvain 이 동작하지 않음. 반드시 `python-louvain`.

```python
import sys
sys.path.append('/path/to/cscore_pkg')    # cscore 폴더의 바로 위 폴더
import cscore as cscr
import numpy as np
import pandas as pd

out_dir  = '/path/to/data_out/'     # CAP, 표 저장 (미리 만들어 둘 것)
plot_dir = '/path/to/figures/'      # 그림 저장 (미리 만들어 둘 것)
```

## 구조

```
cscore/
  cap.py       xi_cor, 유전자 vs 행동, pairwise 행렬, consensus Louvain (CAP)
  pipeline.py  입력 포맷 확인 (check_inputs), 한 번에 돌리기 (run_cscore)
  shift.py     조건 간 CAP 변화 (PCA shift), RGB 색, t-SNE
  modules.py   CAP -> 모듈 (spectral, k 선택, eigengene, MCL)
  compare.py   모듈 비교, within-group 검정, partition 보존, Mantel/CKA
  enrich.py    gene set enrichment, gProfiler, term clustering
  plotting.py  그림 (savepath 를 줄 때만 저장)
```

---

## 기본 분석 (조건 2개 비교)

### 1. 입력 만들기

세 입력 모두 **index = 같은 샘플 ID**.

| 이름 | 형태 | 예 |
|---|---|---|
| `expr` | samples x genes DataFrame (숫자만) | 정규화된 발현값 |
| `beh` | 1 컬럼 DataFrame 또는 Series | 행동 PC1 |
| `conds` | 1 컬럼 DataFrame 또는 Series | 조건 라벨 (0/1, 'yoked'/'trained' 등) |

```python
df = pd.read_csv('my_data.csv', index_col=0)

expr  = df[gene_list];      expr.index  = df.SampleID
beh   = df[['PC1']];        beh.index   = df.SampleID
conds = df[['condition']];  conds.index = df.SampleID

expr2, beh2, conds2 = cscr.check_inputs(expr, beh, conds)
# [check] samples=14, genes=..., conds={0: 7, 1: 7}
```

`check_inputs` 는 샘플 ID 일치, 숫자 여부, NaN, 중복을 확인하고 같은 순서로 맞춰줌.
값이 모든 샘플에서 같은 유전자는 뺌. **이후 단계에는 `expr2`, `beh2`, `conds2` 를 쓸 것.**

### 2. 행동과 관련된 유전자 고르기

```python
res_gene = cscr.gene_vs_target(expr2, beh2)        # 유전자마다 Spearman + xi
res_gene.to_csv(out_dir + 'gene_scores.csv')

sel = cscr.select_genes(res_gene)                  # Z_pv < 0.05
required_genes = ['Fos', 'Arc', 'Npas4']           # 항상 넣을 유전자 (예: candidate)
sel = sorted(set(sel) | {g for g in required_genes if g in expr2.columns})
print('selected genes:', len(sel))
```

### 3. CAP 만들기 (전체 / 조건별)

모든 CAP 은 같은 유전자 목록 (`sel`) 으로 만듦. 대각선은 1.

```python
sets = {'all': expr2.index,
        0: conds2.index[conds2 == 0],
        1: conds2.index[conds2 == 1]}

caps = {}
for label, idx in sets.items():
    long = cscr.pairwise_single(expr2.loc[idx], sel)
    cap = cscr.graph_process_ckpt(long, out_dir + f'CAP_{label}.csv', n_iter=1000)
    cap = cap.reindex(index=sel, columns=sel).fillna(0.0)
    a = cap.values.copy(); np.fill_diagonal(a, 1)
    caps[label] = pd.DataFrame(a, index=sel, columns=sel)
```

- 오래 걸리는 단계. 처음에는 `n_iter=20` 으로 끝까지 한 번 돌려본 뒤 1000 으로.
- `CAP_{label}.convergence.txt` 에 수렴 로그가 남음 (checkpoint 별 상관, split-half 신뢰도).
- 2~3 단계는 `cscr.run_cscore(expr, beh, conds, add_genes=required_genes, out_dir=out_dir)` 한 줄로도 가능.
  결과의 `out['caps']` 가 위 `caps` 와 같음.

### 4. 같은 PCA 공간에서 조건 간 변화 (shift)

전체 CAP 으로 PCA 를 fit 하고, 조건별 CAP 을 같은 축으로 옮긴 뒤 유전자마다 이동을 계산.

```python
cscr.plot_pc_spectrum(cscr.cap_pca_spectrum(caps['all']),
                      savepath=plot_dir + 'pc_spectrum.jpg')     # PC 3개로 충분한지 확인

res_shift = cscr.cap_shift(caps, base=0, target=1, n_components=3)   # target - base
```

- `base`: 기준 조건, `target`: 비교할 조건. 라벨은 `conds` 의 값 그대로.
- `res_shift['shift']` 컬럼
  - `norm`: base -> target 이동 거리
  - `centroid`, `cent_sim`: 전체 이동 방향에 대한 투영값 / cosine
  - `PCdot`, `gene_sim`: base 위치와 target 위치의 내적 / cosine (낮을수록 위치가 많이 바뀜)

### 5. 3D 그림 + 순위

```python
col = cscr.rgb_colors(res_shift['scores'][1], invert='G')    # target 점수로 RGB 색. invert 로 채널 반전
cscr.plot_cap_pca_3d(res_shift, colors=col, highlight=required_genes,
                     label_base='Yoked', label_target='Trained',
                     savepath=plot_dir + 'cap_pca_3d.jpg')
# show=True 를 주면 창에서 마우스로 돌려볼 수 있음 (창을 닫아야 다음 줄로 넘어감)

sh = res_shift['shift']
sh.sort_values('norm', ascending=False).head(10)     # 많이 움직인 순
sh.sort_values('gene_sim').head(10)                  # 위치가 많이 바뀐 순
sh.to_csv(out_dir + 'shift.csv')
```

### 6. t-SNE

```python
ts = cscr.shift_tsne(sh, perplexity=30, random_state=5)      # norm, cent_sim, gene_sim 으로 2D
cscr.plot_shift_tsne(ts, colors=col, highlight=required_genes,
                     savepath=plot_dir + 'tsne.jpg')
# colors 대신 color_col='norm' 처럼 지표 값으로 칠할 수도 있음
```

### 7. MCL 클러스터 + Sankey

조건별 CAP 을 각각 MCL 로 클러스터링하고, 유전자가 base 클러스터에서 target 클러스터로 옮겨간 흐름을 그림.

```python
# inflation 고르기
scan0 = cscr.mcl_scan(caps[0])
scan1 = cscr.mcl_scan(caps[1])
cscr.plot_mcl_scan(scan0, 'Yoked',   savepath=plot_dir + 'mcl_scan_yoked.jpg')
cscr.plot_mcl_scan(scan1, 'Trained', savepath=plot_dir + 'mcl_scan_trained.jpg')

# 정한 inflation 으로 클러스터 (논문은 6.0)
lab0 = cscr.mcl_clusters(caps[0], 6.0, candidates=required_genes)
lab1 = cscr.mcl_clusters(caps[1], 6.0, candidates=required_genes)

# Sankey (html, 브라우저에서 열기)
cscr.plot_cluster_sankey(lab0, lab1, label_base='Yoked', label_target='Trained',
                         savepath=plot_dir + 'sankey.html')
```

- `mcl_clusters` 결과에서 -1 은 어느 클러스터에도 들어가지 않은 유전자.
- Sankey 는 `min_count=2` 미만인 흐름은 생략.

---






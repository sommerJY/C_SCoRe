# cscore

C-SCoRe (Correlation Signal Co-cluster Reduction) analysis package.

## 0. Installation

### Packages needed

| Purpose | Packages (pip names) |
|---|---|
| Core computation | `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`, `joblib`, `tqdm` |
| CAP (Louvain) | `networkx`, `python-louvain` (imported as `community`) |
| Figures | `matplotlib`, `seaborn` |
| MCL, Sankey (step 7) | `markov_clustering`, `plotly` |
| GO analysis (enrich.py) | `gprofiler-official` |

Python 3.9 or later.



### Method 1: Install as a package

```bash
cd cscore
pip install -e .              # Also installs the core packages. No need to reinstall after editing the code.
pip install -e ".[enrich]"    # Use this if you also need MCL, Sankey, and GO
```

### Method 2: Use the folder directly, without installing

Install the required packages yourself, then add the path and import.

```bash
pip install numpy pandas scipy statsmodels scikit-learn joblib tqdm \
            networkx python-louvain matplotlib seaborn \
            markov_clustering plotly gprofiler-official
```

If you use a conda environment, install while that environment is active (`conda activate env_name`).
If you still get `No module named ...`, the package was installed in a different environment.
Check the current environment in Python with `import sys; print(sys.executable)`,
then install with that path, e.g. `/.../bin/python -m pip install package_name`, to be sure.

Note: `pip install community` is a different package and Louvain will not work with it. You must install `python-louvain`.

```python
import sys
sys.path.append('/path/to/cscore_pkg')    # The folder directly above the cscore folder
import cscore as cscr
import numpy as np
import pandas as pd

out_dir  = '/path/to/data_out/'     # Where CAPs and tables are saved (create it beforehand)
plot_dir = '/path/to/figures/'      # Where figures are saved (create it beforehand)
```

## Structure

```
cscore/
  cap.py       xi_cor, gene vs behavior, pairwise matrix, consensus Louvain (CAP)
  pipeline.py  input format check (check_inputs), run everything at once (run_cscore)
  shift.py     CAP change between conditions (PCA shift), RGB colors, t-SNE
  modules.py   CAP -> modules (spectral, choosing k, eigengene, MCL)
  compare.py   module comparison, within-group tests, partition preservation, Mantel/CKA
  enrich.py    gene set enrichment, gProfiler, term clustering
  plotting.py  figures (saved only when savepath is given)
```

---

## Basic analysis (comparing 2 conditions)

### 1. Preparing the inputs

All three inputs must have **the same sample IDs as their index**.

| Name | Format | Example |
|---|---|---|
| `expr` | samples x genes DataFrame (numeric only) | Normalized expression values |
| `beh` | 1-column DataFrame or Series | Behavior PC1 |
| `conds` | 1-column DataFrame or Series | Condition labels (0/1, 'yoked'/'trained', etc.) |

```python
df = pd.read_csv('my_data.csv', index_col=0)

expr  = df[gene_list];      expr.index  = df.SampleID
beh   = df[['PC1']];        beh.index   = df.SampleID
conds = df[['condition']];  conds.index = df.SampleID

expr2, beh2, conds2 = cscr.check_inputs(expr, beh, conds)
# [check] samples=14, genes=..., conds={0: 7, 1: 7}
```

`check_inputs` checks that sample IDs match, values are numeric, and there are no NaNs or duplicates, and it aligns everything to the same order.
Genes with identical values across all samples are removed. **Use `expr2`, `beh2`, and `conds2` in the following steps.**

### 2. Selecting behavior-related genes

```python
res_gene = cscr.gene_vs_target(expr2, beh2)        # Spearman + xi for each gene
res_gene.to_csv(out_dir + 'gene_scores.csv')

sel = cscr.select_genes(res_gene)                  # Z_pv < 0.05
required_genes = ['Fos', 'Arc', 'Npas4']           # Genes to always include (e.g. candidates)
sel = sorted(set(sel) | {g for g in required_genes if g in expr2.columns})
print('selected genes:', len(sel))
```

### 3. Building CAPs (overall / per condition)

All CAPs are built with the same gene list (`sel`). The diagonal is 1.

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

- This step takes a long time. First run through once with `n_iter=20`, then run again with 1000.
- A convergence log is saved to `CAP_{label}.convergence.txt` (correlation per checkpoint, split-half reliability).
- Steps 2 and 3 can also be done in one line with `cscr.run_cscore(expr, beh, conds, add_genes=required_genes, out_dir=out_dir)`.
  The resulting `out['caps']` is the same as `caps` above.

### 4. Change between conditions in the same PCA space (shift)

Fit PCA on the overall CAP, project each condition's CAP onto the same axes, and compute the shift for each gene.

```python
cscr.plot_pc_spectrum(cscr.cap_pca_spectrum(caps['all']),
                      savepath=plot_dir + 'pc_spectrum.jpg')     # Check that 3 PCs are enough

res_shift = cscr.cap_shift(caps, base=0, target=1, n_components=3)   # target - base
```

- `base`: the reference condition. `target`: the condition to compare. Labels are the values in `conds` as they are.
- `res_shift['shift']` columns
  - `norm`: distance moved from base to target
  - `centroid`, `cent_sim`: projection onto / cosine with the overall direction of movement
  - `PCdot`, `gene_sim`: dot product / cosine between the base position and the target position (lower means the position changed more)

### 5. 3D figure + ranking

```python
col = cscr.rgb_colors(res_shift['scores'][1], invert='G')    # RGB colors from the target score. invert flips a channel
cscr.plot_cap_pca_3d(res_shift, colors=col, highlight=required_genes,
                     label_base='Yoked', label_target='Trained',
                     savepath=plot_dir + 'cap_pca_3d.jpg')
# With show=True you can rotate it with the mouse in a window (close the window to move on to the next line)

sh = res_shift['shift']
sh.sort_values('norm', ascending=False).head(10)     # Ordered by how much they moved
sh.sort_values('gene_sim').head(10)                  # Ordered by how much their position changed
sh.to_csv(out_dir + 'shift.csv')
```

### 6. t-SNE

```python
ts = cscr.shift_tsne(sh, perplexity=30, random_state=5)      # 2D from norm, cent_sim, gene_sim
cscr.plot_shift_tsne(ts, colors=col, highlight=required_genes,
                     savepath=plot_dir + 'tsne.jpg')
# Instead of colors, you can also color by a metric value, e.g. color_col='norm'
```

### 7. MCL clusters + Sankey

Cluster each condition's CAP with MCL, and plot the flow of genes from base clusters to target clusters.

```python
# Choosing inflation
scan0 = cscr.mcl_scan(caps[0])
scan1 = cscr.mcl_scan(caps[1])
cscr.plot_mcl_scan(scan0, 'Yoked',   savepath=plot_dir + 'mcl_scan_yoked.jpg')
cscr.plot_mcl_scan(scan1, 'Trained', savepath=plot_dir + 'mcl_scan_trained.jpg')

# Cluster with the chosen inflation (the paper uses 6.0)
lab0 = cscr.mcl_clusters(caps[0], 6.0, candidates=required_genes)
lab1 = cscr.mcl_clusters(caps[1], 6.0, candidates=required_genes)

# Sankey (html, open in a browser)
cscr.plot_cluster_sankey(lab0, lab1, label_base='Yoked', label_target='Trained',
                         savepath=plot_dir + 'sankey.html')
```

- In the `mcl_clusters` result, -1 means a gene that was not assigned to any cluster.
- The Sankey omits flows below `min_count=2`.

---
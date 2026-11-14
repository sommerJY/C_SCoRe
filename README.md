# C-SCoRe

C-SCoRe (Correlation Signal Co-cluster Reduction) analysis package

## Citation

If you use C-SCoRe, please cite:

> Han J, et al. Persistently Increased Expression of PKMzeta and Unbiased Gene Expression Profiles Identify Hippocampal Molecular Traces of a Long-Term Active Place Avoidance Memory and "Shadow" Proteins. *Advanced Science* (2026). https://doi.org/10.1002/advs.202521254



## 0. Installation

### Download

Download the code from GitHub (no `pip install` of the package itself is needed).

```bash
git clone https://github.com/sommerJY/C_SCoRe.git
```

Or use **Code > Download ZIP** on the GitHub page and unzip it.

### Required packages

Tested with Python 3.12.6.

| Purpose | Package (pip name) | Tested version |
|---|---|---|
| Core computation | `numpy` | 2.4.3 |
| | `pandas` | 3.0.1 |
| | `scipy` | 1.17.1 |
| | `statsmodels` | 0.14.6 |
| | `scikit-learn` | 1.8.0 |
| | `joblib` | 1.5.3 |
| | `tqdm` | 4.67.3 |
| CAP (Louvain) | `networkx` | 3.6.1 |
| | `python-louvain` (imported as `community`) | 0.16 |
| Figures | `matplotlib` | 3.10.8 |
| | `seaborn` | 0.13.2 |
| MCL, Sankey (step 7) | `markov_clustering` | 0.0.6 |
| | `plotly` | 6.6.0 |
| GO analysis (enrich.py) | `gprofiler-official` | 1.0.0 |


Install each package individually, for example:

```bash
pip install numpy
pip install pandas
pip install python-louvain
...
```

To match the tested version exactly: `pip install numpy==x.y.z`

Notes
- `pip install community` is a different package and Louvain will not work with it. You must install `python-louvain`.
- If you use a conda environment, install while that environment is active (`conda activate env_name`).
- If you still get `No module named ...`, the package was installed in a different environment.
  Check the current environment in Python with `import sys; print(sys.executable)`,
  then install with that path, e.g. `/.../bin/python -m pip install package_name`.

### Import

```python

import sys
sys.path.append('/path/to/cscore_pkg')    # The folder directly above the cscore folder
sys.path.append('.')
import cscore as cscr
import numpy as np
import pandas as pd

in_dir = 'example/data/'
out_dir = 'example/results/'    
plot_dir = 'example/figure/'     

```

## Structure

```
cscore/
  cap.py       xi_cor, gene vs behavior, pairwise matrix, consensus Louvain (CAP)
  pipeline.py  input format check (check_inputs), run everything at once (run_cscore)
  shift.py     CAP change between conditions (PCA shift), RGB colors, t-SNE
  modules.py   MCL clustering of each condition's CAP
  plotting.py  figures (saved only when savepath is given)
example/
  data/        example input 
  results/    CAPs and tables from the example are saved here
  figure/     figures from the example are saved here
```

---

## Example: run the whole analysis on the example data
The example data is dentate gyrus (DG) bulk RNA-seq, 1 day after training, from the paper.
It has three files, all indexed by the same sample IDs:
`expression.csv` (samples x genes), `behavior.csv` (behavior PC1), and `condition.csv` (0 = Yoked, 1 = Trained).

Run from the top folder of the downloaded repository. Each CAP with `n_iter=1000` takes about 10 min.

```python

import os
import cscore as cscr
import pandas as pd
import numpy as np

os.makedirs(out_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Inputs

beh = pd.read_csv(in_dir+'behavior.csv', index_col = 0)
expr = pd.read_csv(in_dir+'expression.csv', index_col = 0)
conds = pd.read_csv(in_dir+'condition.csv', index_col = 0)

expr, beh, conds = cscr.check_inputs(expr, beh, conds) # checking the format 

# Behavior-related genes 
res = cscr.gene_vs_target(expr, beh)
res.to_csv(out_dir + 'gene_scores.csv')

sel = cscr.select_genes(res)                  # Z_pv < 0.05
print(len(sel))

required_gene_list = [
    "Fos", "Fosl2", "Npas4", "Arc", "Grin1", "Gria1", "Gria2", "Pick1", "Nsf",
    "Numb", "Fmr1", "Camk2a", "Wwc1", "Prkcb", "Prkcz", "Prkci"]

sel = sorted(set(sel) | {g for g in required_gene_list if g in expr.columns})
print('selected genes:', len(sel))

# CAPs (about 10 min each)
sets = {'all': expr.index,
        0: conds.index[conds == 0],
        1: conds.index[conds == 1]}

caps = {}
for label, idx in sets.items():
    long = cscr.pairwise_single(expr.loc[idx], sel)
    cap = cscr.graph_process_ckpt(long, out_dir + f'CAP_{label}.csv', n_iter=1000)
    cap = cap.reindex(index=sel, columns=sel).fillna(0.0)
    a = cap.values.copy(); np.fill_diagonal(a, 1)
    caps[label] = pd.DataFrame(a, index=sel, columns=sel)

# Shift in the shared PCA space
cscr.plot_pc_spectrum(cscr.cap_pca_spectrum(caps['all']), savepath=plot_dir + 'pc_spectrum.jpg')   # check that 3 PCs are enough
res_cscore = cscr.cap_shift(caps, base=0, target=1, n_components=3)            # 1 - 0

col = cscr.rgb_colors(res_cscore['scores'][1], invert='G')                     # colors from target (1) scores, G flipped
cscr.plot_cap_pca_3d(res_cscore, colors=col, highlight=required_gene_list,
                     label_base='Yoked', label_target='Trained', show=True)

sh = res_cscore['shift']
print(sh.sort_values('norm', ascending=False).head(10))                 # moved the most
print(sh.sort_values('gene_sim').head(10))                              # direction changed the most

# t-SNE
ts = cscr.shift_tsne(sh, perplexity=50, random_state=42)
cscr.plot_shift_tsne(ts, colors=col, highlight=required_gene_list, savepath=plot_dir + 'tsne.jpg')
cscr.plot_shift_tsne(ts, color_col='norm', highlight=required_gene_list,
                     savepath=plot_dir + 'tsne_norm.jpg')     # colored by distance moved (viridis)

# MCL: choosing inflation
scan0 = cscr.mcl_scan(caps[0])
scan1 = cscr.mcl_scan(caps[1])
cscr.plot_mcl_scan(scan0, 'Yoked', savepath=plot_dir + 'mcl_scan_yoked.jpg')
cscr.plot_mcl_scan(scan1, 'Trained', savepath=plot_dir + 'mcl_scan_trained.jpg')

# Clusters with the chosen inflation (the paper uses 6.0)
lab0 = cscr.mcl_clusters(caps[0], 6.0, candidates=required_gene_list)
lab1 = cscr.mcl_clusters(caps[1], 6.0, candidates=required_gene_list)

# Sankey (html)
cscr.plot_cluster_sankey(lab0, lab1, label_base='Yoked', label_target='Trained',
                         savepath=plot_dir + 'sankey.html')
```





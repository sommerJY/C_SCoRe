"""그림 함수 모음. 모두 savepath 를 줄 때만 저장."""
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
import seaborn as sns

from ._utils import _abs_pair_corrs, _spearman_cols, _save


def plot_find_modules(results_df, title='', savepath=None):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(results_df['n_modules'], results_df['silhouette'], 'o-', color='darkorange')
    if 'size_filt' in results_df.columns:
        t = results_df['size_filt']
        ax.scatter(results_df.loc[t, 'n_modules'], results_df.loc[t, 'silhouette'],
                   s=80, edgecolor='k', linewidth=1.5, color='black', zorder=3)
        ax.scatter(results_df.loc[~t, 'n_modules'], results_df.loc[~t, 'silhouette'],
                   s=30, edgecolor='k', linewidth=1.0, color='darkorange', zorder=2)
    ax.set_xlabel('k (n_modules)')
    ax.set_ylabel('silhouette')
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_cap_clustermap_real(name, CAP, partition, savepath=None):
    ids = sorted(set(partition.values()))
    cmap = plt.cm.tab20
    cmap_map = {m: cmap(i / max(len(ids), 1)) for i, m in enumerate(ids)}
    rc = pd.Series([cmap_map[partition[g]] for g in CAP.index], index=CAP.index, name='module')
    g = sns.clustermap(CAP, row_colors=rc, col_colors=rc, cmap='bwr', center=0, vmin=-1, vmax=1,
                       xticklabels=False, yticklabels=False, cbar_pos=None,
                       row_cluster=True, col_cluster=True, figsize=(5, 5))
    g.fig.suptitle(name, fontsize=10)
    g.fig.subplots_adjust(top=0.93)
    if savepath:
        g.fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(g.fig)


def plot_eigen_results(name, partition, eigen_df, X, Y, b, eigen_corr,
                       x_id_dict, y_id_dict, x_label='A', y_label='B',
                       min_genes=3, savepath=None, top_print=5):
    """X: 모듈을 정의한 omic 발현, Y: 반대 omic 발현.
    3 panel: intra-module corr / eigen->behavior / eigen->Y"""
    x_pos = {c: i for i, c in enumerate(X.columns)}
    y_cols = list(Y.columns)
    x_vals = X.values
    size = pd.Series(partition).value_counts().to_dict()
    col_pos = {c: i for i, c in enumerate(eigen_df.columns)}
    valid = [int(c.split('_')[1]) for c in eigen_df.columns if size.get(int(c.split('_')[1]), 0) >= min_genes]
    if not valid:
        print('no valid modules')
        return
    x = np.arange(len(valid))
    labels = [f'mod_{m}\n(n={size[m]})' for m in valid]
    bv = b.values if hasattr(b, 'values') else np.asarray(b)

    fig, axes = plt.subplots(3, 1, figsize=(max(4, len(valid) * 1.2), 10))
    intra = []
    for m in valid:
        idx = [x_pos[g] for g, mm in partition.items() if mm == m and g in x_pos]
        intra.append(_abs_pair_corrs(x_vals[:, idx]))
    ax = axes[0]
    ax.violinplot(intra, positions=x, showmedians=True)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('|Spearman|'); ax.set_title(f'Intra-module {x_label} pair correlation')

    rho, pv = [], []
    for m in valid:
        col = f'module_{m}'
        rho.append(eigen_corr['eigen_beh'][col_pos[col]])
        pv.append(stats.spearmanr(eigen_df[col].values, bv)[1])
    ax = axes[1]
    ax.bar(x, rho, color='#B5D4F4', alpha=0.9)
    for i, p in enumerate(pv):
        if p < 0.05:
            ax.text(i, rho[i] + 0.02, '*', ha='center', fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_ylim(-1, 1.1); ax.set_ylabel('Spearman'); ax.set_title('Eigen -> Behavior')

    ax = axes[2]
    data = []
    for m in valid:
        c = np.abs(eigen_corr['eigen_other'][col_pos[f'module_{m}']])
        data.append(c)
        if top_print:
            print(f"mod_{m}:")
            for ti in np.argsort(c)[::-1][:top_print]:
                print(f"  {y_id_dict.get(y_cols[ti], y_cols[ti])}: {c[ti]:.3f}")
    ax.violinplot(data, positions=x, showmedians=True)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.0); ax.set_ylabel('|Spearman|'); ax.set_title(f'Eigen -> {y_label}')
    fig.suptitle(name, fontsize=11)
    plt.tight_layout()
    _save(fig, savepath, 'eigen_results')
    plt.close(fig)


def plot_module_top_features(name, partition, eigen_df, X, Y, eigen_corr,
                             x_id_dict, y_id_dict, x_label='A', y_label='B',
                             top_n=10, min_genes=3, max_cols=5, savepath=None):
    """모듈별 top X (모듈 내부) 와 top Y (반대 omic) bar."""
    x_cols = list(X.columns)
    y_cols = list(Y.columns)
    x_vals = X.values
    col_pos = {c: i for i, c in enumerate(eigen_df.columns)}
    mods = list(eigen_df.columns)
    n_mod = len(mods)
    n_cols = min(n_mod, max_cols)
    n_rows = (n_mod + n_cols - 1) // n_cols
    fig_x, ax_x = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 4 * n_rows))
    fig_y, ax_y = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 4 * n_rows))
    ax_x = np.array(ax_x).flatten()
    ax_y = np.array(ax_y).flatten()
    x_pos = {c: i for i, c in enumerate(x_cols)}
    for a_x, a_y, col in zip(ax_x, ax_y, mods):
        m = int(col.split('_')[1])
        idx = [x_pos[g] for g, mm in partition.items() if mm == m and g in x_pos]
        if len(idx) < min_genes:
            a_x.set_visible(False); a_y.set_visible(False)
            continue
        e = eigen_df[col].values.reshape(-1, 1)
        c = _spearman_cols(x_vals[:, idx], e)[:, 0]
        top = np.argsort(np.abs(c))[::-1][:min(top_n, len(c))]
        names = [x_id_dict.get(x_cols[idx[i]], x_cols[idx[i]]) for i in top]
        vals = c[top]
        a_x.barh(range(len(top)), vals[::-1], color=['#ffe599' if v > 0 else '#e899ff' for v in vals[::-1]], alpha=0.85)
        a_x.set_yticks(range(len(top))); a_x.set_yticklabels(names[::-1], fontsize=8)
        a_x.axvline(0, color='k', linewidth=0.5); a_x.set_xlim(-1, 1)
        a_x.set_title(f'{col} -> {x_label}', fontsize=9)

        oc = np.nan_to_num(eigen_corr['eigen_other'][col_pos[col]], nan=0.0)
        topy = np.argsort(np.abs(oc))[::-1][:top_n]
        namesy = [y_id_dict.get(y_cols[i], y_cols[i]) for i in topy]
        valsy = oc[topy]
        a_y.barh(range(len(topy)), valsy[::-1], color=['#ffe599' if v > 0 else '#e899ff' for v in valsy[::-1]], alpha=0.85)
        a_y.set_yticks(range(len(topy))); a_y.set_yticklabels(namesy[::-1], fontsize=8)
        a_y.axvline(0, color='k', linewidth=0.5); a_y.set_xlim(-1, 1)
        a_y.set_title(f'{col} -> {y_label}', fontsize=9)
    for a in ax_x[n_mod:]:
        a.set_visible(False)
    for a in ax_y[n_mod:]:
        a.set_visible(False)
    fig_x.suptitle(f'{name} - top {x_label} per module', fontsize=11)
    fig_y.suptitle(f'{name} - top {y_label} per module', fontsize=11)
    fig_x.tight_layout(); fig_y.tight_layout()
    _save(fig_x, savepath, f'Top_{x_label}_per_module')
    _save(fig_y, savepath, f'Top_{y_label}_per_module')
    plt.close(fig_x)
    plt.close(fig_y)


def plot_module_module_heatmap(name, mm, part_a, part_b, label_a='A', label_b='B', savepath=None):
    corr, pval = mm['module_corr'], mm['pval_matrix']
    A_ids, B_ids = mm['module_ids_a'], mm['module_ids_b']
    As = pd.Series(part_a).value_counts().to_dict()
    Bs = pd.Series(part_b).value_counts().to_dict()
    gl = [f'{i}\n(n={As.get(int(i.split("_")[1]), 0)})' for i in A_ids]
    pl = [f'{i}\n(n={Bs.get(int(i.split("_")[1]), 0)})' for i in B_ids]
    fig, ax = plt.subplots(figsize=(max(4, len(B_ids) * 0.6), max(3, len(A_ids) * 0.6)))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    for i in range(len(A_ids)):
        for j in range(len(B_ids)):
            ax.text(j, i, f"{corr[i, j]:.2f}{'*' if pval[i, j] < 0.05 else ''}", ha='center', va='center', fontsize=7)
    ax.set_xticks(range(len(B_ids))); ax.set_xticklabels(pl, rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
    ax.set_yticks(range(len(A_ids))); ax.set_yticklabels(gl, fontsize=8)
    ax.set_xlabel(f'{label_b} module'); ax.set_ylabel(f'{label_a} module'); ax.set_title(name)
    fig.colorbar(im, ax=ax, label='Spearman')
    plt.tight_layout()
    _save(fig, savepath, 'module_module_heatmap')
    plt.close(fig)


def compare_pvals(df, savepath=None):
    x = -np.log10(df['hyper_pval'].clip(lower=1e-300))
    y = -np.log10(df['perm_pval'].clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, color='#4C72B0')
    lim = max(x.max(), y.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'r--', linewidth=0.8)
    ax.set_xlabel('-log10(hyper_pval)'); ax.set_ylabel('-log10(perm_pval)')
    ax.set_title('Hypergeometric vs Permutation p-values')
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_module_module_heatmap_with_geneset(name, mm, part_a, part_b,
                                         set_df_a, set_df_b, set_label='gene set', label_a='A', label_b='B',
                                         savepath=None):
    corr, pval = mm['module_corr'], mm['pval_matrix']
    A_ids, B_ids = mm['module_ids_a'], mm['module_ids_b']
    As = pd.Series(part_a).value_counts().to_dict()
    Bs = pd.Series(part_b).value_counts().to_dict()
    gl = [f'{i}\n(n={As.get(int(i.split("_")[1]), 0)})' for i in A_ids]
    pl = [f'{i}\n(n={Bs.get(int(i.split("_")[1]), 0)})' for i in B_ids]

    gmap = dict(zip(set_df_a['module'], -np.log10(set_df_a['hyper_fdr'].clip(lower=1e-300))))
    pmap = dict(zip(set_df_b['module'], -np.log10(set_df_b['hyper_fdr'].clip(lower=1e-300))))
    gv = [gmap.get(i, 0) for i in A_ids]
    pv = [pmap.get(i, 0) for i in B_ids]

    gov = dict(zip(set_df_a['module'], set_df_a['n_target_in_module']))
    pov = dict(zip(set_df_b['module'], set_df_b['n_target_in_module']))
    gn = [gov.get(i, 0) for i in A_ids]
    pn = [pov.get(i, 0) for i in B_ids]

    fig = plt.figure(figsize=(max(6, len(B_ids) * 0.9) + 1.5, max(5, len(A_ids) * 0.9) + 1.5))
    grid = gridspec.GridSpec(2, 2, width_ratios=[len(B_ids), 1.2], height_ratios=[1, len(A_ids)],
                             hspace=0.05, wspace=0.05)
    a_top = fig.add_subplot(grid[0, 0])
    a_main = fig.add_subplot(grid[1, 0])
    a_right = fig.add_subplot(grid[1, 1])
    im = a_main.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    for i in range(len(A_ids)):
        for j in range(len(B_ids)):
            a_main.text(j, i, f"{corr[i, j]:.2f}{'*' if pval[i, j] < 0.05 else ''}", ha='center', va='center', fontsize=7)
    a_main.set_xticks(range(len(B_ids))); a_main.set_xticklabels(pl, rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
    a_main.set_yticks(range(len(A_ids))); a_main.set_yticklabels(gl, fontsize=8)
    a_main.set_xlabel(f'{label_b} module'); a_main.set_ylabel(f'{label_a} module')

    a_top.bar(range(len(B_ids)), pv, color='#4C72B0')
    a_top.axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=0.8)
    ymax = max(pv) if pv else 0
    off = max(ymax * 0.03, 0.05)
    for j, (v, n) in enumerate(zip(pv, pn)):
        a_top.text(j, v + off, f'overlap={n}', ha='center', va='bottom', fontsize=6)
    a_top.set_ylim(0, ymax * 1.15 + off)
    a_top.set_xticks(range(len(B_ids))); a_top.set_xticklabels([])
    a_top.set_xlim(-0.5, len(B_ids) - 0.5)
    a_top.set_ylabel(f'-log10(FDR)\n{set_label}', fontsize=7)
    a_top.set_title(f'{name}: eigen_corr + {set_label} enrichment', fontsize=10)

    a_right.barh(range(len(A_ids)), gv, color='#55A868')
    a_right.axvline(-np.log10(0.05), color='red', linestyle='--', linewidth=0.8)
    xmax = max(gv) if gv else 0
    offr = max(xmax * 0.03, 0.05)
    for i, (v, n) in enumerate(zip(gv, gn)):
        a_right.text(v + offr, i, f'{n}', ha='left', va='center', fontsize=6)
    a_right.set_xlim(0, xmax * 1.25 + offr)
    a_right.set_yticks(range(len(A_ids))); a_right.set_yticklabels([])
    a_right.set_ylim(-0.5, len(A_ids) - 0.5)
    a_right.invert_yaxis()
    a_right.set_xlabel(f'-log10(FDR)\n{set_label}', fontsize=7)
    fig.colorbar(im, ax=[a_main, a_top, a_right], label='Spearman', shrink=0.6, pad=0.02)
    _save(fig, savepath, 'heatmap_with_geneset')
    plt.close(fig)


def plot_module_module_heatmap_with_geneset2(name, mm, part_a, part_b,
                                          set_df_a, set_df_b, set_label='gene set', label_a='A', label_b='B',
                                          savepath=None):
    # 회의 결정: dot 크기 = overlap 비율, 색 = correlation 강도, annotation 숫자는 그대로
    # 주의: mm 에 'overlap_matrix' 가 없으면 dot 크기가 전부 0 (compute_module_module_correlations 는 이 키를 만들지 않음)
    corr = mm['module_corr']
    pval = mm['pval_matrix']
    A_ids, B_ids = mm['module_ids_a'], mm['module_ids_b']
    As = pd.Series(part_a).value_counts().to_dict()
    Bs = pd.Series(part_b).value_counts().to_dict()
    gl = [f'{i}\n(n={As.get(int(i.split("_")[1]), 0)})' for i in A_ids]
    pl = [f'{i}\n(n={Bs.get(int(i.split("_")[1]), 0)})' for i in B_ids]
    gmap = dict(zip(set_df_a['module'], -np.log10(set_df_a['hyper_fdr'].clip(lower=1e-300))))
    pmap = dict(zip(set_df_b['module'], -np.log10(set_df_b['hyper_fdr'].clip(lower=1e-300))))
    gv = [gmap.get(i, 0) for i in A_ids]
    pv = [pmap.get(i, 0) for i in B_ids]
    gov = dict(zip(set_df_a['module'], set_df_a['n_target_in_module']))
    pov = dict(zip(set_df_b['module'], set_df_b['n_target_in_module']))
    gn = [gov.get(i, 0) for i in A_ids]
    pn = [pov.get(i, 0) for i in B_ids]
    overlap_mat = mm['overlap_matrix'] if 'overlap_matrix' in mm else None
    overlap_ratio = np.zeros_like(corr, dtype=float)
    if overlap_mat is not None:
        for i in range(len(A_ids)):
            g_size = As.get(int(A_ids[i].split("_")[1]), 1)
            for j in range(len(B_ids)):
                overlap_ratio[i, j] = overlap_mat[i, j] / g_size if g_size > 0 else 0
    n_g, n_p = len(A_ids), len(B_ids)
    main_w = max(6, n_p * 0.9)
    main_h = max(5, n_g * 0.9)
    fig = plt.figure(figsize=(main_w + 3.2, main_h + 2.2))
    gs_grid = GridSpec(2, 3, figure=fig,
                       width_ratios=[main_w, 1.6, 0.3],
                       height_ratios=[1.6, main_h],
                       wspace=0.08, hspace=0.08)
    ax = fig.add_subplot(gs_grid[1, 0])
    a_top = fig.add_subplot(gs_grid[0, 0], sharex=ax)
    a_right = fig.add_subplot(gs_grid[1, 1], sharey=ax)
    cax = fig.add_subplot(gs_grid[1, 2])
    a_leg = fig.add_subplot(gs_grid[0, 1:3])
    a_leg.axis('off')
    sizes = 500 * overlap_ratio.flatten()
    norm = plt.Normalize(vmin=-1, vmax=1)
    for i in range(n_g):
        for j in range(n_p):
            c = corr[i, j]
            color = plt.cm.RdBu_r(norm(c))
            ax.scatter(j, i, s=sizes[i * n_p + j], color=color, edgecolors='k', alpha=0.7)
            text_str = f"{c:.2f}{'*' if pval[i, j] < 0.05 else ''}\n{round(overlap_ratio[i, j] * 100)}%"
            ax.text(j, i, text_str, ha='center', va='center', fontsize=7, color='black',
                    path_effects=[pe.withStroke(linewidth=1.2, foreground=(1, 1, 1, 0.55))])
    ax.set_xticks(range(n_p)); ax.set_xticklabels(pl, rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
    ax.set_yticks(range(n_g)); ax.set_yticklabels(gl, fontsize=8)
    ax.set_xlabel(f'{label_b} module'); ax.set_ylabel(f'{label_a} module')
    ax.set_xlim(-0.5, n_p - 0.5)
    ax.set_ylim(n_g - 0.5, -0.5)
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
    fig.colorbar(sm, cax=cax, label='Spearman')
    legend_ratios = [0.2, 0.5, 0.8]
    legend_handles = [plt.scatter([], [], s=500 * r, color='#555', edgecolors='k', alpha=0.7)
                      for r in legend_ratios]
    a_leg.legend(legend_handles, [f'{int(r * 100)}% overlap' for r in legend_ratios],
                 scatterpoints=1, frameon=True, labelspacing=1,
                 title='Overlap ratio (dot size)', loc='center')
    a_top.bar(range(n_p), pv, color='#4C72B0')
    a_top.axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=0.8)
    ymax = max(pv) if pv else 0
    off = max(ymax * 0.03, 0.05)
    for j, (v, n) in enumerate(zip(pv, pn)):
        a_top.text(j, v + off, f'overlap={n}', ha='center', va='bottom', fontsize=6)
    a_top.set_ylim(0, ymax * 1.15 + off)
    a_top.tick_params(axis='x', labelbottom=False)
    a_top.set_ylabel(f'-log10(FDR)\n{set_label}', fontsize=7)
    a_top.set_title(f'{name}: eigen_corr(dot color) + overlap(dotsize)', fontsize=11)
    a_right.barh(range(n_g), gv, color='#55A868')
    a_right.axvline(-np.log10(0.05), color='red', linestyle='--', linewidth=0.8)
    xmax = max(gv) if gv else 0
    offr = max(xmax * 0.03, 0.05)
    for i, (v, n) in enumerate(zip(gv, gn)):
        a_right.text(v + offr, i, f'{n}', ha='left', va='center', fontsize=6)
    a_right.set_xlim(0, xmax * 1.25 + offr)
    a_right.tick_params(axis='y', labelleft=False)
    a_right.set_xlabel(f'-log10(FDR)\n{set_label}', fontsize=7)
    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_module_preservation_bars(name, df, ref_label='reference', test_label='test', savepath=None):
    df = df.copy()
    df['mod_num'] = df['module'].apply(lambda m: int(m.split('_')[1]))
    df = df.sort_values('mod_num')
    y = np.arange(len(df))
    h = 0.35
    fig, ax = plt.subplots(figsize=(6, max(4, len(df) * 0.6)))
    ax.barh(y + h / 2, df['ref_coherence'], h, label=ref_label, color='#4C72B0')
    ax.barh(y - h / 2, df['test_coherence'], h, label=test_label, color='#C44E52')
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m} (n={n})" for m, n in zip(df['module'], df['n_genes'])], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |intra-module Spearman|')
    ax.set_title(name)
    ax.legend()
    plt.tight_layout()
    _save(fig, savepath, 'module_preservation_bars')
    plt.close(fig)


def plot_preservation_barh(df, title='', figsize=(2, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    d = df.reset_index(drop=True)
    d = d.iloc[::-1]
    ax.barh(d['module'], d['preserve_rate'])
    ax.set_title(title, fontsize=8)
    plt.tight_layout()
    plt.close(fig)


def plot_partition_crosstab(name, partition_a, partition_b, a_label='A', b_label='B', savepath=None):
    common = set(partition_a) & set(partition_b)
    am = sorted(set(partition_a[g] for g in common))
    bm = sorted(set(partition_b[g] for g in common))
    ai = {m: i for i, m in enumerate(am)}
    bi = {m: i for i, m in enumerate(bm)}
    mat = np.zeros((len(am), len(bm)))
    for g in common:
        mat[ai[partition_a[g]], bi[partition_b[g]]] += 1
    rs = mat.sum(axis=1, keepdims=True)
    norm = np.divide(mat, rs, out=np.zeros_like(mat), where=rs != 0)
    fig, ax = plt.subplots(figsize=(max(4, len(bm) * 0.7), max(3, len(am) * 0.7)))
    im = ax.imshow(norm, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    for i in range(len(am)):
        for j in range(len(bm)):
            if mat[i, j] > 0:
                ax.text(j, i, f'{int(mat[i, j])}', ha='center', va='center', fontsize=8)
    ax.set_xticks(range(len(bm))); ax.set_xticklabels([f'{b_label}_m{m}' for m in bm], rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
    ax.set_yticks(range(len(am))); ax.set_yticklabels([f'{a_label}_m{m}\n(n={int(rs[i, 0])})' for i, m in enumerate(am)], fontsize=8)
    ax.set_xlabel(f'to {b_label} module'); ax.set_ylabel(f'original {a_label} module')
    ax.set_title(f'{name}: gene reassignment {a_label} -> {b_label}')
    fig.colorbar(im, ax=ax, label=f'Proportion of {a_label} module')
    plt.tight_layout()
    _save(fig, savepath, 'partition_crosstab')
    plt.close(fig)


def plot_pair_dots(tab, n_shared, title, savepath=None, set_r=None, set_p=None, set_label='gene set',
                   label_a='A', label_b='B'):
    """칸 색 = 그룹 내 |Spearman|, * = maxT p<0.05, 검은 테두리 = robust, n = 겹친 유전자 수."""
    key = lambda s: int(s.split('_')[1])
    rm = sorted(tab['module_a'].unique(), key=key)
    pm = sorted(tab['module_b'].unique(), key=key)
    nr = tab.drop_duplicates('module_a').set_index('module_a')['n_a'].to_dict()
    npro = tab.drop_duplicates('module_b').set_index('module_b')['n_b'].to_dict()
    use_f = set_r is not None and set_p is not None
    w, h = 2.6 + 0.8 * len(pm), 1.8 + 0.5 * len(rm)
    if use_f:
        fig = plt.figure(figsize=(w + 1.8, h + 1.5))
        gsp = fig.add_gridspec(2, 2, width_ratios=[len(pm), 1.6], height_ratios=[1.6, len(rm)],
                               hspace=0.05, wspace=0.05)
        ax = fig.add_subplot(gsp[1, 0])
        a_top = fig.add_subplot(gsp[0, 0], sharex=ax)
        a_right = fig.add_subplot(gsp[1, 1], sharey=ax)
    else:
        fig, ax = plt.subplots(figsize=(w, h))
    norm = plt.Normalize(0, 1)
    for _, rr in tab.iterrows():
        x, y = pm.index(rr['module_b']), rm.index(rr['module_a'])
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=plt.cm.YlOrRd(norm(abs(rr['within_r']))),
                                   edgecolor='k' if rr['robust'] else 'white', linewidth=2.5 if rr['robust'] else 0.5))
        ax.text(x, y, f"{abs(rr['within_r']):.2f}{'*' if rr['p_maxT'] < 0.05 else ''}\nn={rr['n_overlap']}",
                ha='center', va='center', fontsize=6.5)
    ax.set_xticks(range(len(pm)))
    ax.set_xticklabels([f'{m}\n(n={npro[m]})' for m in pm], fontsize=8)
    ax.set_yticks(range(len(rm)))
    ax.set_yticklabels([f'{m}\n(n={nr[m]})' for m in rm], fontsize=8)
    ax.set_xlim(-0.5, len(pm) - 0.5)
    ax.set_ylim(len(rm) - 0.5, -0.5)
    ax.set_xlabel(f'{label_b} module')
    ax.set_ylabel(f'{label_a} module')
    if use_f:
        def bars(df, mods):
            d = df.set_index('module')
            v = [-np.log10(max(d.loc[m, 'hyper_fdr'], 1e-300)) if m in d.index else 0 for m in mods]
            n = [int(d.loc[m, 'n_target_in_module']) if m in d.index else 0 for m in mods]
            return v, n
        pv, pn = bars(set_p, pm)
        gv, gn = bars(set_r, rm)
        thr = -np.log10(0.05)
        a_top.bar(range(len(pm)), pv, color='#4C72B0')
        a_top.axhline(thr, color='red', ls='--', lw=0.8)
        off = max(max(pv) * 0.03, 0.05)
        for j, (v, n) in enumerate(zip(pv, pn)):
            a_top.text(j, v + off, str(n), ha='center', va='bottom', fontsize=6)
        a_top.set_ylim(0, max(max(pv) * 1.2, thr * 1.3))
        a_top.tick_params(axis='x', labelbottom=False)
        a_top.set_ylabel(f'-log10(FDR)\n{set_label}', fontsize=7)
        a_right.barh(range(len(rm)), gv, color='#55A868')
        a_right.axvline(thr, color='red', ls='--', lw=0.8)
        offr = max(max(gv) * 0.03, 0.05)
        for i, (v, n) in enumerate(zip(gv, gn)):
            a_right.text(v + offr, i, str(n), ha='left', va='center', fontsize=6)
        a_right.set_xlim(0, max(max(gv) * 1.3, thr * 1.3))
        a_right.tick_params(axis='y', labelleft=False)
        a_right.set_xlabel(f'-log10(FDR)\n{set_label}', fontsize=7)
        a_top.set_title(f'{title}\ncolor: within-group |Spearman| (* maxT p<0.05, black box = robust)\n'
                        f'n = shared genes (among {n_shared}); bar number = {set_label} genes in module', fontsize=8)
    else:
        ax.set_title(f'{title}\ncolor: within-group |Spearman| (* maxT p<0.05, black box = robust)\n'
                     f'n = shared genes in both modules (among {n_shared} detected in both datasets)', fontsize=8)
        fig.colorbar(plt.cm.ScalarMappable(cmap='YlOrRd', norm=norm), ax=ax, label='|Spearman|', shrink=0.6)
        plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=250, bbox_inches='tight')
    plt.close(fig)


def sunburst_html(df, path, values, title, savepath):
    """plotly sunburst (interactive html). pip install plotly (png 저장은 kaleido 필요)"""
    try:
        import plotly.express as px
    except ImportError:
        print('plotly 가 없어서 sunburst 생략: pip install plotly kaleido')
        return None
    fig = px.sunburst(df, path=path, values=values, title=title)
    fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))
    fig.write_html(savepath)
    return fig


# ----------------------------------------------------------------------
# 조건 간 CAP 변화 (shift.py 결과용)
# ----------------------------------------------------------------------
def plot_pc_spectrum(spec, savepath=None):
    """shift.cap_pca_spectrum 결과: eigenvalue / 설명분산 + 누적"""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].plot(spec['PC'], spec['eigenvalue'], 'o-')
    axes[0].set_xticks(spec['PC']); axes[0].set_xlabel('PC'); axes[0].set_ylabel('Eigenvalue')
    axes[0].grid(axis='y')
    axes[1].bar(spec['PC'], spec['var_ratio'], alpha=0.6)
    axes[1].plot(spec['PC'], spec['cum_ratio'], 'r--', marker='o', label='Cumulative')
    axes[1].set_xticks(spec['PC']); axes[1].set_xlabel('PC'); axes[1].set_ylabel('Variance Ratio')
    axes[1].grid(axis='y')
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    return fig

def plot_cap_pca_3d(res, colors=None, highlight=None, label_base=None, label_target=None,
                    elev=30, azim=90, savepath=None, show=False):
    """shift.cap_shift 결과: base(왼쪽), target(오른쪽) 의 CAP PCA 점수를 같은 축 범위로 나란히 3D.
    colors: gene -> 색 (None 이면 target 점수 기준 RGB), highlight: 이름 표시할 유전자 목록
    show=True: 창을 띄워서 마우스로 돌려볼 수 있음 (창을 닫아야 다음 줄로 넘어감)"""
    from .shift import rgb_colors
    Sa, Sb = res['scores'][res['base']], res['scores'][res['target']]
    cols = list(Sa.columns[:3])
    colors = rgb_colors(Sb) if colors is None else pd.Series(colors).reindex(Sa.index).fillna('#999999')
    hl = [g for g in (highlight or []) if g in Sa.index]
    vr = res['var_ratio']
    both = pd.concat([Sa[cols], Sb[cols]])
    lims = [(both[c].min(), both[c].max()) for c in cols]
    fig = plt.figure(figsize=(16, 6))
    for k, (S, title) in enumerate([(Sa, label_base or str(res['base'])),
                                    (Sb, label_target or str(res['target']))]):
        ax = fig.add_subplot(1, 2, k + 1, projection='3d')
        rest = [g for g in S.index if g not in hl]
        ax.scatter(S.loc[rest, cols[0]], S.loc[rest, cols[1]], S.loc[rest, cols[2]],
                   c=colors.loc[rest].tolist(), marker='o', s=50, alpha=0.5, linewidth=0)
        if hl:
            ax.scatter(S.loc[hl, cols[0]], S.loc[hl, cols[1]], S.loc[hl, cols[2]],
                       c=colors.loc[hl].tolist(), marker='d', s=120, alpha=0.8, edgecolor='black', linewidth=1)
            for g in hl:
                ax.text(*S.loc[g, cols].values, g, color='k', fontsize=8)
        ax.set_title(title)
        for i, setlab in enumerate([ax.set_xlabel, ax.set_ylabel, ax.set_zlabel]):
            setlab(f'PC{i + 1} ({vr[i] * 100:.1f}%)')
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
        ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    if show:                 # 추가
        plt.show()           # 추가
    return fig


def plot_shift_tsne(tsne_df, color_col='norm', colors=None, highlight=None, cmap='viridis',
                    size_col=None, title=None, savepath=None):
    """shift.shift_tsne 결과 2D 그림.
    color_col: 색으로 쓸 지표 (colors 를 주면 그 색을 그대로 사용, 예: rgb_colors 결과)
    size_col : 점 크기로 쓸 지표 (None 이면 고정)"""
    d = tsne_df
    hl = [g for g in (highlight or []) if g in d.index]
    rest = [g for g in d.index if g not in hl]
    size = (lambda idx: d.loc[idx, size_col] * 10) if size_col else (lambda idx: 40)
    fig, ax = plt.subplots(figsize=(7, 5))
    if colors is not None:
        c = pd.Series(colors).reindex(d.index).fillna('#999999')
        kw = dict()
        c_rest, c_hl = c.loc[rest].tolist(), c.loc[hl].tolist()
    else:
        norm = plt.Normalize(d[color_col].min(), d[color_col].max())
        kw = dict(cmap=cmap, norm=norm)
        c_rest, c_hl = d.loc[rest, color_col], d.loc[hl, color_col]
    sc = ax.scatter(d.loc[rest, 'comp1'], d.loc[rest, 'comp2'], c=c_rest, s=size(rest),
                    marker='o', alpha=0.5, linewidth=0, **kw)
    if hl:
        ax.scatter(d.loc[hl, 'comp1'], d.loc[hl, 'comp2'], c=c_hl, s=size(hl) if size_col else 80,
                   marker='d', alpha=0.8, linewidth=0, **kw)
        for g in hl:
            ax.text(d.loc[g, 'comp1'], d.loc[g, 'comp2'], g, fontsize=9, ha='center')
    if colors is None:
        fig.colorbar(sc, ax=ax, label=color_col)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or f't-SNE (color: {color_col if colors is None else "custom"})')
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    return fig





def plot_mcl_scan(scan, title='', savepath=None):
    """modules.mcl_scan 결과: inflation 별 intra 유사도(파랑) / silhouette(빨강), 숫자 = 클러스터 수"""
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.set_title(title)
    ax1.set_xlabel('Inflation')
    ax1.set_ylabel('Intra-cluster similarity', color='blue')
    ax1.plot(scan['inflation'], scan['avg_intra_similarity'], '-o', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Silhouette', color='red')
    ax2.plot(scan['inflation'], scan['silhouette_score'], '-s', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    for _, r in scan.iterrows():
        for ax, col in [(ax1, 'avg_intra_similarity'), (ax2, 'silhouette_score')]:
            if not np.isnan(r[col]):
                ax.annotate(f"{int(r['num_clusters'])}", (r['inflation'], r[col]),
                            textcoords='offset points', xytext=(-5, 5), ha='center', fontsize=8)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    return fig


def plot_cluster_sankey(labels_base, labels_target, label_base='base', label_target='target',
                        min_count=2, savepath=None):
    """base 클러스터 -> target 클러스터로 유전자가 옮겨간 흐름 (plotly html).
    labels_*: gene -> cluster Series (modules.mcl_clusters 결과). min_count 미만 흐름은 생략."""
    import plotly.graph_objects as go
    from matplotlib.colors import to_rgba
    both = pd.DataFrame({'b': labels_base, 't': labels_target}).dropna()
    mv = both.groupby(['b', 't']).size().reset_index(name='count')
    mv = mv[mv['count'] >= min_count].copy()
    mv['b'] = 'L' + mv['b'].astype(int).astype(str)
    mv['t'] = 'R' + mv['t'].astype(int).astype(str)
    nodes = list(mv['b'].unique()) + list(mv['t'].unique())
    cmap = plt.get_cmap('tab20')
    base_ids = sorted(mv['b'].unique(), key=lambda s: int(s[1:]))
    col = {n: tuple(int(255 * v) for v in to_rgba(cmap(i % 20))[:3]) for i, n in enumerate(base_ids)}
    node_col = [f'rgba{col[n] + (1,)}' if n.startswith('L') else 'rgba(128,128,128,1)' for n in nodes]
    link_col = [f'rgba{col[n] + (0.4,)}' for n in mv['b']]
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color='black', width=0.5), label=nodes, color=node_col),
        link=dict(source=[nodes.index(x) for x in mv['b']], target=[nodes.index(x) for x in mv['t']],
                  value=mv['count'].tolist(), color=link_col))])
    fig.update_layout(title=f'{label_base} (L) -> {label_target} (R)')
    if savepath:
        fig.write_html(savepath)
    return fig
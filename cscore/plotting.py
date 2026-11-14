"""Plotting functions for the two-condition C-SCoRe analysis. All of them save only when savepath is given."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# CAP change between conditions (for shift.py results)
# ----------------------------------------------------------------------
def plot_pc_spectrum(spec, savepath=None):
    """shift.cap_pca_spectrum result: eigenvalue / explained variance + cumulative"""
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
    """shift.cap_shift result: 3D plots of the CAP PCA scores of base (left) and target (right) side by side, with the same axis ranges.
    colors: gene -> color (if None, RGB based on target scores), highlight: list of genes to label by name
    show=True: opens a window you can rotate with the mouse (close the window to move on to the next line)"""
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
    if show:                 # added
        plt.show()           # added
    return fig


def plot_shift_tsne(tsne_df, color_col='norm', colors=None, highlight=None, cmap='viridis',
                    size_col=None, title=None, savepath=None):
    """2D plot of the shift.shift_tsne result.
    color_col: metric used for color (if colors is given, those colors are used as they are, e.g. the rgb_colors result)
    size_col : metric used for point size (fixed if None)"""
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
    """modules.mcl_scan result: intra similarity (blue) / silhouette (red) per inflation, numbers = number of clusters"""
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
    """Flow of genes moving from base clusters to target clusters (plotly html).
    labels_*: gene -> cluster Series (modules.mcl_clusters result). Flows below min_count are omitted."""
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
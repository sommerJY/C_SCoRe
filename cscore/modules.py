"""CAP -> modules: concat similarity, spectral clustering, choosing k, eigengene."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score


# ----------------------------------------------------------------------
# concat similarity -> spectral clustering
# ----------------------------------------------------------------------
def make_concat_sim(*caps):
    """Concatenate several CAPs (gene x gene) side by side and compute gene-gene correlation."""
    idx = caps[0].index
    aligned = [c.loc[idx, idx] for c in caps]
    concat = pd.concat(aligned, axis=1)
    sim = np.corrcoef(concat.values)
    return pd.DataFrame(sim, index=idx, columns=idx)


def find_modules_real(affinity, min_modules=3, max_modules=15, verbose=True):
    aff = np.nan_to_num(affinity.values.copy(), nan=0.0)
    np.clip(aff, 0, 1, out=aff)
    np.fill_diagonal(aff, 1.0)
    dist = 1 - aff
    np.fill_diagonal(dist, 0)
    results = []
    for k in range(min_modules, max_modules + 1):
        sc = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
        labels = sc.fit_predict(aff)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(dist, labels, metric='precomputed')
        sizes = sorted([int(v) for v in pd.Series(labels).value_counts().values], reverse=True)
        results.append((k, sil, labels, sizes))
        if verbose:
            print(f"k={k}: sil={sil:.3f}, sizes={sizes}")
    df = pd.DataFrame([(k, sil) for k, sil, _, _ in results], columns=['n_modules', 'silhouette'])
    return results, df


def scan_caps(caps, min_modules=3, max_modules=15, min_size=10, title=''):
    """List of CAPs -> concat sim -> scan over k. (General version that works without a project-specific load_cap)"""
    sim = make_concat_sim(*caps)
    results, df = find_modules_real(sim, min_modules, max_modules)
    df['size_filt'] = [min(r[3]) >= min_size for r in results]
    return dict(sim=sim, results=results, results_df=df, title=title)


def pick_partition(results, affinity, chosen_k):
    labels = [r[2] for r in results if r[0] == chosen_k][0]
    partition = {g: int(labels[i]) for i, g in enumerate(affinity.index)}
    print(f"chosen_k={chosen_k}, n_modules={len(set(partition.values()))}")
    return partition, labels


def partition_from_scan(scan, k):
    lab = [r[2] for r in scan['results'] if r[0] == k][0]
    return {g: int(l) for g, l in zip(scan['sim'].index, lab)}


# ----------------------------------------------------------------------
# eigengene
# ----------------------------------------------------------------------
def compute_module_representative(data, mod_idx, method='pc1'):
    sub = data[:, mod_idx]
    if method == 'mean':
        return sub.mean(axis=1)
    if sub.shape[1] == 1:
        return sub.squeeze()
    if method == 'pc1':
        pca = PCA(n_components=1)
        r = pca.fit_transform(sub).squeeze()
        if pca.components_[0, 0] < 0:
            r = -r
        return r
    if method == 'signed_mean':
        pca = PCA(n_components=1)
        pca.fit(sub)
        s = np.sign(pca.components_[0])
        s[s == 0] = 1
        return (sub * s[None, :]).mean(axis=1)
    raise ValueError(method)


def compute_eigengenes_multi(partition, expr_dict, method='pc1', min_genes=3, verbose=True):
    """expr_dict: {label: DataFrame(samples x features)} -> {label: DataFrame(samples x modules)}"""
    cols = list(list(expr_dict.values())[0].columns)
    col_pos = {c: i for i, c in enumerate(cols)}
    idx_by_mod = {}
    for m in sorted(set(partition.values())):
        mod_genes = [g for g, mm in partition.items() if mm == m and g in col_pos]
        if len(mod_genes) < min_genes:
            continue
        idx_by_mod[m] = [col_pos[g] for g in mod_genes]
        if verbose:
            print(f"  module_{m}: {len(mod_genes)} features")
    out = {}
    for name, df in expr_dict.items():
        eg = {f'module_{m}': compute_module_representative(df.values, idx, method)
              for m, idx in idx_by_mod.items()}
        out[name] = pd.DataFrame(eg, index=df.index)
    return out


def module_eigengenes(expr, partition, min_genes=3, method='pc1'):
    """Representative value per module. Genes are z-scored, then PC1 (or a sign-aligned mean, 'zmean').
    The sign is fixed so that it correlates positively with the module's mean expression (just a convention, not an absolute direction).
    Returns: (samples x 'module_k' DataFrame, QC table: explained variance, kME sign consistency)"""
    cols = set(expr.columns)
    out, qc = {}, []
    for m in sorted(set(partition.values())):
        gs = [g for g, mm in partition.items() if mm == m and g in cols]
        if len(gs) < min_genes:
            continue
        X = expr[gs].astype(float).values
        sd = X.std(0)
        X = (X - X.mean(0)) / np.where(sd == 0, 1, sd)
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        pc1 = u[:, 0] * s[0]
        if method == 'zmean':
            sg = np.sign(vt[0])
            sg[sg == 0] = 1
            rep = (X * sg).mean(1)
        else:
            rep = pc1
        if np.nan_to_num(np.corrcoef(rep, X.mean(1))[0, 1]) < 0:
            rep = -rep
        with np.errstate(invalid='ignore'):
            kme = np.array([np.corrcoef(rep, X[:, i])[0, 1] for i in range(X.shape[1])])
        out[f'module_{m}'] = rep
        qc.append(dict(module=f'module_{m}', n_genes=len(gs), var_explained=s[0] ** 2 / np.sum(s ** 2),
                       frac_pos_kME=float(np.nanmean(kme > 0)), median_kME=float(np.nanmedian(kme))))
    return pd.DataFrame(out, index=expr.index), pd.DataFrame(qc)


signed_eigengenes = module_eigengenes      # compatible with the existing meeting code


# ----------------------------------------------------------------------
# Choosing k
# ----------------------------------------------------------------------
def _affinity(sim):
    aff = np.nan_to_num(sim.values.copy(), nan=0.0)
    np.clip(aff, 0, 1, out=aff)
    np.fill_diagonal(aff, 1.0)
    return aff


def k_table(scan, expr, n_rep=10, frac=0.8, seed=0, stab=True, min_size=10, red_thr=0.8,
            stab_min=0.6, sil_tol=0.02):
    """One table of evidence for each k.
    silhouette / minimum module size / stability under 80% gene subsampling (ARI) / eigengene redundancy (number of pairs with |r|>red_thr).
    suggest: among k that pass the filters, the smallest k whose silhouette is at least (best - sil_tol) (parsimony).
    expr: expression of the data the modules were built from (samples x genes, all samples). This is only a suggestion; the final k is chosen by a person."""
    sim = scan['sim']
    aff = _affinity(sim)
    n = len(aff)
    rng = np.random.default_rng(seed)
    rows = []
    for k, sil, labels, sizes in scan['results']:
        part = {g: int(l) for g, l in zip(sim.index, labels)}
        eg, qc = module_eigengenes(expr, part)
        if eg.shape[1] > 1:
            C = np.abs(np.nan_to_num(np.corrcoef(eg.values.T)))
            np.fill_diagonal(C, 0)
        else:
            C = np.zeros((1, 1))
        row = dict(k=k, silhouette=sil, min_size=min(sizes), max_size=max(sizes),
                   med_var_expl=qc['var_explained'].median() if len(qc) else np.nan,
                   max_eig_corr=C.max(), n_redundant_pairs=int((np.triu(C, 1) > red_thr).sum()))
        if stab:
            full = np.asarray(labels)
            aris = []
            for _ in range(n_rep):
                s = np.sort(rng.choice(n, int(frac * n), replace=False))
                lab = SpectralClustering(n_clusters=k, affinity='precomputed',
                                         random_state=int(rng.integers(1_000_000))).fit_predict(aff[np.ix_(s, s)])
                aris.append(adjusted_rand_score(full[s], lab))
            row.update(stab_ari=float(np.mean(aris)), stab_sd=float(np.std(aris)))
        rows.append(row)
    df = pd.DataFrame(rows).set_index('k')
    ok = (df['min_size'] >= min_size) & (df['n_redundant_pairs'] == 0)
    if stab:
        ok &= df['stab_ari'] >= stab_min
    df['pass'] = ok
    df['suggest'] = False
    if ok.any():
        best = df.loc[ok, 'silhouette'].max()
        df.loc[df.index[(df['silhouette'] >= best - sil_tol) & ok].min(), 'suggest'] = True
    return df


def redundancy_by_k(scan, expr, groups=None, red_thr=0.8):
    """Eigengene pair correlation for each k (sign kept). If groups is given, group means are subtracted and only within-group variation is used.
    r > +red_thr : module pairs moving together in the same direction = candidates for over-splitting (n_pos_redundant_w)
    r < -red_thr : opposite direction = may be the two ends of one axis. Not over-splitting (n_neg_w)"""
    sim = scan['sim']
    g = None if groups is None else np.asarray(groups)
    rows = []
    for k, sil, labels, sizes in scan['results']:
        part = {gn: int(l) for gn, l in zip(sim.index, labels)}
        eg, _ = module_eigengenes(expr, part)
        V = eg.values.copy()
        if g is not None:
            for u in np.unique(g):
                m = g == u
                V[m] = V[m] - V[m].mean(0)
        if V.shape[1] > 1:
            C = np.nan_to_num(np.corrcoef(V.T))
            np.fill_diagonal(C, 0)
        else:
            C = np.zeros((1, 1))
        iu = np.triu_indices(C.shape[0], 1)
        v = C[iu]
        rows.append(dict(k=k, max_pos_r_w=v.max() if len(v) else 0.0, min_neg_r_w=v.min() if len(v) else 0.0,
                         n_pos_redundant_w=int((v > red_thr).sum()), n_neg_w=int((v < -red_thr).sum())))
    return pd.DataFrame(rows).set_index('k')


def refilter_k_table(ktab, red, min_size=10, stab_min=0.6, sil_tol=0.02):
    """Attach signed within-group redundancy to an existing k_table and recompute pass / suggest (only positive redundancy is penalized)"""
    drop = [c for c in ['max_eig_corr_w', 'n_redundant_w', 'max_pos_r_w', 'min_neg_r_w', 'n_pos_redundant_w', 'n_neg_w'] if c in ktab]
    df = ktab.drop(columns=drop).join(red)
    ok = (df['min_size'] >= min_size) & (df['n_pos_redundant_w'] == 0)
    if 'stab_ari' in df:
        ok &= df['stab_ari'] >= stab_min
    df['pass'] = ok
    df['suggest'] = False
    if ok.any():
        best = df.loc[ok, 'silhouette'].max()
        df.loc[df.index[(df['silhouette'] >= best - sil_tol) & ok].min(), 'suggest'] = True
    return df


# ----------------------------------------------------------------------
# partition utilities
# ----------------------------------------------------------------------
def module_symbols(partition, id2sym):
    out = {}
    for g, m in partition.items():
        s = id2sym.get(g, g)
        if isinstance(s, str) and s:
            out.setdefault(m, set()).add(s)
    return {m: sorted(v) for m, v in out.items()}


def partition_to_symbol(partition, id_dict):
    return {id_dict[g]: m for g, m in partition.items() if g in id_dict}


def module_genes(partition, id_dict, module):
    return [id_dict.get(g, g) for g, m in partition.items() if m == module]


def find_gene_module(partition, id_dict, symbol):
    ids = [g for g in partition if id_dict.get(g, g) == symbol]
    for g in ids:
        print(f"{symbol} ({g}) -> module_{partition[g]}")
    return {g: partition[g] for g in ids}


def candidate_clusters(partition, id_dict, candidates, verbose=True):
    """Which module each candidate gene is in."""
    rows = []
    for g, m in partition.items():
        s = id_dict.get(g, g)
        if s in candidates:
            rows.append(dict(id=g, gene=s, module=m))
    df = pd.DataFrame(rows).sort_values('module') if rows else pd.DataFrame(columns=['id', 'gene', 'module'])
    if verbose and len(df):
        for m, grp in df.groupby('module'):
            print(f"  module_{m}: {', '.join(sorted(grp['gene']))}")
        missing = [c for c in candidates if c not in set(df['gene'])]
        if missing:
            print(f"  not found: {', '.join(missing)}")
    return df




# ----------------------------------------------------------------------
# MCL clustering (one CAP per condition, as in the paper)
# ----------------------------------------------------------------------
def _mcl_labels(cap, inflation):
    import markov_clustering as mc
    genes = list(cap.index)
    adj = cap.fillna(0).values
    clusters = mc.get_clusters(mc.run_mcl(adj, inflation=inflation))
    labels = {g: -1 for g in genes}
    for cid, cl in enumerate(clusters):
        for i in cl:
            labels[genes[i]] = cid
    return pd.Series(labels, name='cluster'), clusters


def mcl_scan(cap, inflations=None, verbose=True):
    """MCL results for each inflation: number of clusters / within-cluster cosine similarity / silhouette.
    inflations default: 2.5 to 14.5 (step 0.5, same as the paper)"""
    from sklearn.metrics.pairwise import cosine_similarity
    if inflations is None:
        inflations = [round(i * 0.5, 2) for i in range(5, 30)]
    cap = cap.fillna(0)
    sim = cosine_similarity(cap.values)
    dist = np.nan_to_num(np.maximum(0, 1 - sim), nan=1.0)
    rows = []
    for infl in inflations:
        lab, clusters = _mcl_labels(cap, infl)
        intra = []
        for cl in clusters:
            if len(cl) < 2:
                continue
            idx = list(cl)
            sub = sim[np.ix_(idx, idx)]
            intra.append(sub[np.triu_indices(len(idx), k=1)].mean())
        arr = lab.values
        u, c = np.unique(arr, return_counts=True)
        valid = [x for x in u[c > 1] if x != -1]
        mask = np.isin(arr, valid)
        sil = (silhouette_score(dist[mask][:, mask], arr[mask], metric='precomputed')
               if len(np.unique(arr[mask])) > 1 else np.nan)
        row = dict(inflation=infl, num_clusters=len(clusters),
                   avg_intra_similarity=np.mean(intra) if intra else np.nan, silhouette_score=sil)
        if verbose:
            print(f"[infl={infl:.2f}] clusters={row['num_clusters']} | intra={row['avg_intra_similarity']:.3f} "
                  f"| silhouette={sil:.3f}")
        rows.append(row)
    return pd.DataFrame(rows)


def mcl_clusters(cap, inflation, candidates=None):
    """Run MCL with the chosen inflation. Returns: gene -> cluster Series (-1 = not in any cluster)"""
    lab, clusters = _mcl_labels(cap, inflation)
    print(f"genes={len(lab)}, assigned={int((lab != -1).sum())}, clusters={len(clusters)}")
    if candidates:
        print(lab[lab.index.isin(candidates)].sort_values().to_string())
    return lab
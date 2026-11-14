"""CAP -> clusters with MCL (one CAP per condition, as in the paper)."""
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

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
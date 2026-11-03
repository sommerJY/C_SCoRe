# function separation



def xi_cor(x, y, method='dense'):
    n = len(x)
    rank_x = stats.rankdata(x, method=method)
    sorted_y = np.asarray(y)[np.argsort(rank_x)]
    r_sorted_y = stats.rankdata(sorted_y, method=method)
    A = np.sum(np.abs(np.diff(r_sorted_y)))
    # O(N^2)에서 O(N log N)으로 최적화: y_j >= y_i 개수 계산
    l = n + 1 - stats.rankdata(y, method='max') 
    D = 2 * np.sum(l * (n - l))
    return float(np.clip(1 - (n * A) / D, 0.0, 1.0))




def scaling_model(x, c): 
    return x[1] / (1 - c / x[0])


obs_res = []
for i in range(1, 11): 
    N = 10 * i
    x_linear = np.linspace(0, 10, N)
    y_linear = x_linear
    observed_xicor = xi_cor(x_linear, y_linear, 'dense')  
    pearsonr = stats.pearsonr(x_linear, y_linear)[0]
    obs_res.append((N, observed_xicor, pearsonr))

N_values = np.array([x[0] for x in obs_res])
observed_xicor_values = np.array([x[1] for x in obs_res])
answer = np.array([x[2] for x in obs_res])

# 여기서 계산된 popt가 전역 변수가 되어 get_new_xi에서 사용됨
popt, pcov = curve_fit(scaling_model, (N_values, observed_xicor_values), answer)

print(f"[*] Calibration Complete. popt value: {popt[0]}") # 2.7705122949082126





def get_new_xi(n, old_xi):
    new = scaling_model((n, old_xi), popt[0])
    return np.clip(new, 0, 1)


def pval_zeta_rayleigh(xi, rho, n):
    # Z-scoring 
    z_xi = xi / np.sqrt(0.4 / n) # xi ~ N(0,2/5n) , statistical z scoring 
    z_rho = rho * np.sqrt(n - 1)  # Spearman ~ N(0,1/(n−1)), t-dist z scoring 
    zeta_stat = np.sqrt(z_xi**2 + z_rho**2)
    # Rayleigh 분포의 survival function (1 - CDF)
    p_value = np.exp(-zeta_stat**2 / 2)
    return zeta_stat, p_value # only for pvalue calculation 


# --- Parallel Workers (Unified logic) ---
def compute_metrics(X_re, Y_re, n, method='dense', symmetric=False): # 바닥 붙어서 안움직이는 애들
    if np.ptp(X_re) == 0 or np.ptp(Y_re) == 0 : # 최댓값 최솟값 차이가 0 이면 
        return {'SCOR2': 0, 'PCOR2': 0, 'XI_new': 0, 'Z_pv': 1.0, 'SIGMA': 0}
    pc, pp = stats.pearsonr(X_re, Y_re)
    sc, sp = stats.spearmanr(X_re, Y_re)
    pc2, sc2 = abs(pc), abs(sc)
    if symmetric:
        ox = max(xi_cor(X_re, Y_re, method), xi_cor(Y_re, X_re, method))
    else:
        ox = xi_cor(X_re, Y_re, method)
    nx = get_new_xi(n, ox)
    z_xi = max(0, ox) / np.sqrt(0.4 / n)
    z_z, zp = pval_zeta_rayleigh(ox, sc2, n)
    return {
        'PCOR': pc, 'P_pv': pp, 'SCOR': sc, 'S_pv': sp, 
        'XI_ori': ox, 'XI_new': nx, 'XI_pv': 1 - stats.norm.cdf(z_xi), 
        'SIGMA': np.sqrt(sc2**2 + nx**2), 'Z_pv': zp, 
        'PCOR2': pc2, 'SCOR2': sc2
    }


def PC_sub_parallel(MY_LOG, gene, PC_ref, method='dense'):
    res = compute_metrics(MY_LOG[gene], MY_LOG[PC_ref], len(MY_LOG), method, symmetric=False)
    res['ID'] = gene; return res

def gene_pair_sub_parallel(MY_LOG, g1, g2, method='dense'):
    res = compute_metrics(MY_LOG[g1], MY_LOG[g2], len(MY_LOG), method, symmetric=True)
    res.update({'geneA':g1, 'geneB':g2, 'MAX':max(res['XI_new'], res['SCOR2'])}); return res

def Parallel_corr_all(RNA_DF, PROT_DF, samples, g1, g2, method='dense'):
    res = compute_metrics(RNA_DF.loc[samples, g1], PROT_DF.loc[samples, g2], len(samples), method, symmetric=True)
    res.update({'geneA':g1, 'geneB':g2, 'MAX':max(res['XI_new'], res['SCOR2'])}); return res

# --- Pipeline Functions ---
def filter_data_sub_treat(data, sub, treat): 
    return data[(data['sub'] == sub) & (data['treatment2'] == treat)]

def PC_based_parallel_do(Name, filtered_data, gene_avail_list, PC_ref, gene_match):
    results = Parallel(n_jobs=10)(delayed(PC_sub_parallel)(filtered_data, g, PC_ref) for g in tqdm(gene_avail_list))
    df = pd.DataFrame(results).merge(gene_match, on='ID', how='left')
    df.to_csv(f"{processed_dir}03.Pall_PC_{Name}.csv"); 
    return df

def run_pairwise_pipeline(filtered_data, Name, label, selected_genes, candy_genes):
    with_c = list(np.unique(list(selected_genes) + list(candy_genes)))
    res = Parallel(n_jobs=10)(delayed(gene_pair_sub_parallel)(filtered_data, g1, g2) for g1, g2 in tqdm(combinations(with_c, 2)))
    pd.DataFrame(res).to_csv(f"{processed_dir}03.G_G_{Name}_{label}.csv")
    return np.unique(selected_genes), np.unique(with_c)

def run_pairwise_pipeline_RNA_PROT(RNA_f, PROT_f, Name, label, R_sel, P_sel, R_can, P_can, samples):
    R_wc, P_wc = list(np.unique(list(R_sel)+list(R_can))), list(np.unique(list(P_sel)+list(P_can)))
    res = Parallel(n_jobs=10)(delayed(Parallel_corr_all)(RNA_f, PROT_f, samples, g1, g2) for g1, g2 in tqdm(product(R_wc, P_wc)))
    pd.DataFrame(res).to_csv(f"{processed_dir}03.G_G_{Name}_{label}.csv")
    return np.unique(R_sel), np.unique(P_sel), np.unique(R_wc), np.unique(P_wc)

def draw_double_scatter(df, NAME, candy):
    cmap = LinearSegmentedColormap.from_list("custom", ['#6A00A8',"#00ffff","#fff678"], N=256)
    v_min = df["Z_pv"].min()
    norm = mcolors.Normalize(vmin=v_min, vmax=0.05) if v_min < 0.05 else mcolors.Normalize(0.05, 0.05)
    df["color"] = df["Z_pv"].apply(lambda x: mcolors.to_hex(cmap(norm(x)))) if v_min < 0.05 else '#fff678'
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    for i, (y_col, y_lab) in enumerate([('XI_new', s_xicor), ('PCOR2', '|'+s_pcc+'|')]):
        sns.scatterplot(x='SCOR2', y=y_col, data=df, hue="color", palette={c:c for c in df["color"].unique()}, ax=axes[i], legend=False, s=10, alpha=0.6)
        c_pts = df[df['ID'].isin(candy.ID)]
        for _, r in c_pts.iterrows():
            axes[i].scatter(r.SCOR2, r[y_col], color=r.color, edgecolors='orangered', s=15)
            axes[i].text(r.SCOR2, r[y_col], candy[candy.ID==r.ID]['gene'].item(), fontsize=4, ha='center', va='center')
        axes[i].set(xlim=(0,1), ylim=(0,1), xlabel='|'+s_scc+'|', ylabel=y_lab, aspect='equal')
    for ext in ['png','pdf','eps','tiff']: plt.savefig(f"{plotpath}03.{NAME}.scatter_all.{ext}", dpi=300, bbox_inches='tight')


def clustering_all(Name, treat, iteration = 1000):
    for comp, conds in [('Comp1', ['ALL','YOKED','TRAINED']), ('Comp2', ['PM','GM'])]:
        for c in conds:
            fname = f'03.G_G_{Name}.{comp}_{c}.csv' if c != 'ALL' else f'03.G_G_{Name}.Comp1_ALL.csv'
            if os.path.exists(processed_dir + fname):
                graph_process(pd.read_csv(processed_dir + fname, index_col=0), f'03.Louvain.{Name}.{comp}.{c}.csv', iteration, 8)
    if treat == 'wt':
        for c in ['PM','GM']:
            graph_process(pd.read_csv(f"{processed_dir}03.G_G_{Name}.Comp3_TrainOnly.{c}.csv", index_col=0), f"03.Louvain.{Name}.Comp3.TrainOnly.{c}.csv", iteration, 8)



def run_pairwise_only(mode, cond, sub, R_ids, P_ids):
    r_f, p_f = filter_data_sub_treat(RNA_TOT, sub, cond), filter_data_sub_treat(PROT_TOT, sub, cond)
    print(f"\n[PAIRWISE] Mode: {mode} | {cond}_{sub} (R:{len(R_ids)}, P:{len(P_ids)})")
    if mode == 'SINGLE':
        for target, data, ids, can in [('RNA', r_f, R_ids, list(RNA_candy.ID)), 
                                       ('PROT', p_f, P_ids, list(Prot_candy.ID))]:
            base = f'{DATE}.{target}.{cond}.{sub}'
            print(f"Running {base}.Comp1 ALL")
            run_pairwise_pipeline(data, f'{base}.Comp1', 'ALL', ids, can)
            for c_k in ['yo','tr','PM','GM'] + (['TrainOnly.PM','TrainOnly.GM'] if cond=='wt' else []):
                comp = 'Comp1' if c_k in ['yo','tr'] else ('Comp2' if c_k in ['PM','GM'] else 'Comp3')
                print(f"Running {base}.{comp}.{condi_rules[c_k]['label']}")
                run_pairwise_pipeline(condi_rules[c_k]['func'](data), f'{base}.{comp}', condi_rules[c_k]['label'], ids, can)
    else: # RNA_PROT 모드
        base = f'{DATE}.RNA_PROT.{cond}.{sub}'
        print(f"Running {base}.Comp1 ALL")
        run_pairwise_pipeline_RNA_PROT(r_f, p_f, f'{base}.Comp1', 'ALL', R_ids, P_ids, list(RNA_candy.ID), list(Prot_candy.ID), list(r_f.index))
        for c_k in ['yo','tr','PM','GM'] + (['TrainOnly.PM','TrainOnly.GM'] if cond=='wt' else []):
            comp = 'Comp1' if c_k in ['yo','tr'] else ('Comp2' if c_k in ['PM','GM'] else 'Comp3')
            f_r, f_p = condi_rules[c_k]['func'](r_f), condi_rules[c_k]['func'](p_f)
            print(f"Running {base}.{comp}.{condi_rules[c_k]['label']}")
            run_pairwise_pipeline_RNA_PROT(f_r, f_p, f'{base}.{comp}', condi_rules[c_k]['label'], R_ids, P_ids, list(RNA_candy.ID), list(Prot_candy.ID), list(f_r.index))



def run_clustering_only(mode, cond, sub, iteration=1000):
    print(f"[CLUSTERING] Mode: {mode} | {cond}_{sub} (Iter: {iteration})")
    targets = ['RNA', 'PROT'] if mode == 'SINGLE' else ['RNA_PROT']
    for target in targets:
        base = f'{DATE}.{target}.{cond}.{sub}'
        for cp in ['Comp2', 'Comp3']:
            src = f"{processed_dir}03.G_G_{base}.Comp1_ALL.csv"
            dst = f"{processed_dir}03.G_G_{base}.{cp}_ALL.csv"
            if os.path.exists(src):
                os.system(f"cp {src} {dst}")
        clustering_all(base, cond, iteration)


# from the graph making 
def process_seed(seed, G, node_index, n):
    local_co_matrix = np.zeros((n, n), dtype=np.uint16) # 속도 
    partition = community_louvain.best_partition(G, random_state=seed)    
    cluster_to_nodes = defaultdict(list)
    for node, cid in partition.items():
        cluster_to_nodes[cid].append(node)    
    for cluster_nodes in cluster_to_nodes.values():
        for i in range(len(cluster_nodes)):
            for j in range(i + 1, len(cluster_nodes)):
                idx1 = node_index[cluster_nodes[i]]
                idx2 = node_index[cluster_nodes[j]]
                local_co_matrix[idx1, idx2] += 1
                local_co_matrix[idx2, idx1] += 1
    n_clusters = len(set(partition.values()))
    return local_co_matrix, n_clusters


def graph_process(data, out_filename, n_iter=1000, n_jobs=8):
    G = nx.Graph()
    for _, r in data.iterrows(): 
        G.add_edge(r['geneA'], r['geneB'], weight=r['SIGMA'])
    nodes = list(G.nodes())
    n = len(nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_seed)(s, G, node_index, n) for s in tqdm(range(n_iter), desc=f"Clustering {out_filename}")
    )
    aggregate = np.zeros((n, n), dtype=np.uint32)
    cluster_counts = [] 
    for local_mat, csize in results:
        aggregate += local_mat
        cluster_counts.append(csize)
    unique_sizes = sorted(set(cluster_counts))
    count_by_size = [(size, cluster_counts.count(size)) for size in unique_sizes]
    print(f"\n[Result] Cluster count distribution: {count_by_size}")
    co_matrix_df = pd.DataFrame(aggregate / n_iter, index=nodes, columns=nodes)
    co_matrix_df.to_csv(os.path.join(processed_dir, out_filename))
    return co_matrix_df



def get_selected_genes(target, sub):
    wt_pc = pd.read_csv(f"{processed_dir}03.Pall_PC_{DATE}.{target}.wt.{sub}.csv")
    ko_pc = pd.read_csv(f"{processed_dir}03.Pall_PC_{DATE}.{target}.ko.{sub}.csv")
    wt_sel = wt_pc[wt_pc['Z_pv'] < 0.05]['ID']
    ko_sel = ko_pc[ko_pc['Z_pv'] < 0.05]['ID']
    combined_genes = np.unique(list(wt_sel) + list(ko_sel))
    return list(combined_genes)





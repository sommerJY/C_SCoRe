

import cscore as cscr
import importlib
# importlib.reload(cscr.shift)
# importlib.reload(cscr.plotting)
# importlib.reload(cscr)          # 마지막에: cscr.cap_shift 같은 최상위 이름 갱신
import pandas as pd 
import numpy as np 
print(hasattr(cscr, 'pipeline'), hasattr(cscr, 'run_cscore'))

now_path = '/Users/jiyeonhan/Dropbox/NYU_LIFE/Memory_ML/'
out_dir = '/Users/jiyeonhan/Dropbox/NYU_LIFE/Memory_ML/01.APA_PKMZ/APA_PKMZ/data_all/'
plot_dir = '/Users/jiyeonhan/Dropbox/NYU_LIFE/Memory_ML/01.APA_PKMZ/APA_PKMZ/figures_all/'




[1 day]
one_day_data_WT = pd.read_csv('/Users/jiyeonhan/Dropbox/NYU_LIFE/Memory_ML/01.APA_PKMZ/APA_PKMZ/data/03.EXP_PC1_merge.DG.csv', index_col = 0)

gene_list = one_day_data_WT.columns.tolist()
gene_list = gene_list[0:16467]

expr = one_day_data_WT[gene_list]
expr.index = one_day_data_WT.RNAseqID
expr.index.name = ''

beh = one_day_data_WT[['PC1']]
beh.columns = ['eigenBehavior']
beh.index = one_day_data_WT.RNAseqID
beh.index.name = ''

conds = one_day_data_WT[['training_class']]
conds.columns = ['condition']
conds.index = one_day_data_WT.RNAseqID
conds.index.name = ''



expr2, beh2, conds2 = cscr.check_inputs(expr, beh, conds)


# 1. original analysis
res = cscr.gene_vs_target(expr2, beh2) # 여기까진 확인함
res.to_csv(out_dir + 'gene_scores_day1.csv')

sel = cscr.select_genes(res)                 # Z_pv < 0.05
print(len(sel))

required_gene_list = ["Fos", "Fosl2", "Npas4", "Arc", "Grin1", "Gria1", "Gria2", "Pick1", "Nsf",
                  "Numb", "Fmr1", "Camk2a", "Wwc1", "Prkcb", "Prkcz", "Prkci"]

sel = sorted(set(sel) | {g for g in required_gene_list if g in expr2.columns})
print('selected genes:', len(sel))


# each CAP 10min 
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



cscr.plot_pc_spectrum(cscr.cap_pca_spectrum(caps['all']), savepath=plot_dir + 'pc_spectrum.jpg')   # PC 3개로 충분한지 확인
res_cscore = cscr.cap_shift(caps, base=0, target=1, n_components=3)            # 1 - 0

col = cscr.rgb_colors(res_cscore['scores'][1], invert='G')                     # target(1) 점수 기준, G 반전
cscr.plot_cap_pca_3d(res_cscore, colors=col, highlight=required_gene_list,
                     label_base='Yoked', label_target='Trained', show=True)

sh = res_cscore['shift']                                                       # 먼저 만들고
print(sh.sort_values('norm', ascending=False).head(10))                 # 많이 움직인 순
print(sh.sort_values('gene_sim').head(10))                              # 방향이 많이 바뀐 순

ts = cscr.shift_tsne(sh, perplexity=50, random_state=42)                 # t-SNE 계산

cscr.plot_shift_tsne(ts, colors=col, highlight=required_gene_list, savepath=plot_dir + 'tsne.jpg')      # t-SNE 도 같은 색으로
cscr.plot_shift_tsne(ts, color_col='norm', highlight=required_gene_list,
                     savepath=plot_dir + 'tsne_norm.jpg')     # 이동 거리로 색칠 (viridis)


# MCL: inflation 고르기
scan0 = cscr.mcl_scan(caps[0])
scan1 = cscr.mcl_scan(caps[1])
cscr.plot_mcl_scan(scan0, 'Yoked', savepath=plot_dir + 'mcl_scan_yoked.jpg')
cscr.plot_mcl_scan(scan1, 'Trained', savepath=plot_dir + 'mcl_scan_trained.jpg')

# 정한 inflation 으로 클러스터 (논문은 6.0)
lab0 = cscr.mcl_clusters(caps[0], 6.0, candidates=required_gene_list)
lab1 = cscr.mcl_clusters(caps[1], 6.0, candidates=required_gene_list)

# Sankey (html)
cscr.plot_cluster_sankey(lab0, lab1, label_base='Yoked', label_target='Trained',
                         savepath=plot_dir + 'sankey.html')







##################### 1 week analysis with FMR dataset #####################
fmr_data_path = '/Users/jiyeonhan/Dropbox/NYU_LIFE/Memory_ML/02.FMR1_Project/FMR1_Project/data/processed/'

RNA_Normalised_pre = pd.read_csv(fmr_data_path+'02.RNA_Normalised_will_0815.csv', index_col = 0) # 15779

GTF_info = pd.read_csv(fmr_data_path + '00.RNA_ID_biotype.csv', index_col=0)
GTF_info_filter = GTF_info[GTF_info['biotype'] == 'protein_coding'].reset_index(drop=True)
RNA_Normalised = RNA_Normalised_pre[RNA_Normalised_pre.ID.isin(list(GTF_info_filter.ID))].reset_index(drop=True)

pca_res_BOTH = pd.read_csv(fmr_data_path + '01.top5_T3T4_pcadf_0814.csv', index_col=0)
pca_res_BOTH['PC1'] = pca_res_BOTH['PC1'] * -1
pca_res_BOTH_4 = pca_res_BOTH[pca_res_BOTH['trialNum'] == 4]
pca_res_BOTH_4.index = pca_res_BOTH_4['new_ID']


RNA_all_columns = list(RNA_Normalised.columns)
RNA_all_samples = RNA_all_columns[0:RNA_all_columns.index('ID')]
RNA_info = pd.DataFrame({
    'RNA_ID': RNA_all_samples,
    'sub': [a.split('_')[4] for a in RNA_all_samples],
    'training': [a.split('_')[1] for a in RNA_all_samples],
    'treatment': [a.split('_')[0] for a in RNA_all_samples],
    'num2': [a.split('_')[2] for a in RNA_all_samples],
    'sex': [a.split('_')[3] for a in RNA_all_samples],
})
RNA_info['treatment2'] = RNA_info['treatment'].apply(lambda a: a.replace('lc', 'wt'))
RNA_info['behav_ID'] = RNA_info['treatment2'] + '_' + RNA_info['training'] + '_' + RNA_info['num2'] + '_' + RNA_info['sex']
RNA_gene_ID = RNA_Normalised[['ID', 'gene']]


# 1. RNA 샘플 x 유전자 (단백질과 겹치는 샘플만이 아니라 RNA 샘플 전체)
RNA = RNA_Normalised.set_index('ID')[RNA_all_samples].T          # samples x Ensembl ID

# 유전자 이름(symbol)으로 바꾸기: 1 day 분석과 같은 방식으로 쓰려고
id2sym = RNA_gene_ID.dropna().set_index('ID')['gene'].to_dict()
RNA = RNA.rename(columns=id2sym)
RNA = RNA.loc[:, ~RNA.columns.duplicated()]                      # 같은 symbol 이 여러 개면 첫 번째만

# 2. 샘플 정보 + 행동 PC1
info = RNA_info.set_index('RNA_ID').join(pca_res_BOTH_4[['PC1']], on='behav_ID')
def _new_tr_tr(r):
    if r['treatment2'] == 'wt':
        return 'wt_PM' if r['PC1'] < 1 else 'wt_GM'
    return 'ko_PM' if r['PC1'] < 0 else 'ko_GM'

info['new_tr_tr'] = info.apply(_new_tr_tr, axis=1)
info['yo_pm_gm'] = info['training'].where(info['training'] == 'yo',
                                          info['new_tr_tr'].str.split('_').str[1])   # 'yo' / 'PM' / 'GM'


def make_exp(treat, sub, cond_col='training'):
    idx = info.index[(info['treatment2'] == treat) & (info['sub'] == sub)]
    idx = idx[info.loc[idx, 'PC1'].notna()]                      # 행동값 없는 샘플 제외
    expr  = RNA.loc[idx]
    beh   = info.loc[idx, ['PC1']].rename(columns={'PC1': 'eigenBehavior'})
    conds = info.loc[idx, [cond_col]].rename(columns={cond_col: 'condition'})
    return cscr.check_inputs(expr, beh, conds)

groups = [(t, s) for t in ['wt', 'ko'] for s in ['UB', 'LB']]

EXP_yotr  = {f'{t}_{s}': make_exp(t, s, 'training')  for t, s in groups}   # yo / tr
EXP_pmgm  = {f'{t}_{s}': make_exp(t, s, 'new_tr_tr') for t, s in groups}   # PM / GM (yoked 포함)
EXP_yopg  = {f'{t}_{s}': make_exp(t, s, 'yo_pm_gm')  for t, s in groups}   # yo / PM / GM (PM, GM 은 trained 만)


n_iter = 1000            # 처음엔 20 으로 끝까지 한 번 돌려보기
min_n = 4                # 조건 샘플이 이보다 적으면 건너뜀


# 1. 유전자 선정: sub 마다 wt, ko 합집합 + candidate (Comp 와 상관없이 같음)
sel = {}
for s in ['UB', 'LB']:
    genes = {}
    for t in ['wt', 'ko']:
        e, b, c = EXP_yotr[f'{t}_{s}']
        r = cscr.gene_vs_target(e, b)
        r.to_csv(fmr_data_path + f'gene_scores_{t}_{s}.csv')
        genes[t] = set(cscr.select_genes(r))
        print(f'  {t}_{s}: {len(genes[t])}')
    union = genes['wt'] | genes['ko']
    sel[s] = sorted(union | set(required_gene_list))
    print(f'{s}: wt {len(genes["wt"])} + ko {len(genes["ko"])} '
          f'(shared {len(genes["wt"] & genes["ko"])}) -> union {len(union)} -> +candidate {len(sel[s])}')

  wt_UB: 725
  ko_UB: 313
UB: wt 725 + ko 313 (shared 12) -> union 1026 -> +candidate 1039 
  wt_LB: 610
  ko_LB: 253
LB: wt 610 + ko 253 (shared 12) -> union 851 -> +candidate 863 



# 2. CAP 하나 만들기 (이미 저장된 게 있으면 읽어옴)
def build_cap(e, idx, genes, path):
    if os.path.exists(path):
        cap = pd.read_csv(path, index_col=0)
    else:
        long = cscr.pairwise_single(e.loc[idx], genes)
        cap = cscr.graph_process_ckpt(long, path, n_iter=n_iter)
    cap = cap.reindex(index=genes, columns=genes).fillna(0.0)
    a = cap.values.copy(); np.fill_diagonal(a, 1)
    return pd.DataFrame(a, index=genes, columns=genes)

DATE = '20261106'          # 실행 날짜 (또는 'H_1106_test' 같은 태그)
import os 

# 3. 그룹 x Comp 별 CAP
COMPS = {'Comp1': EXP_yotr, 'Comp2': EXP_pmgm, 'Comp3': EXP_yopg}
CAPS = {}

for t, s in groups:
    key = f'{t}_{s}'
    e = EXP_yotr[key][0]
    genes = [g for g in sel[s] if g in e.columns]
    cap_all = build_cap(e, e.index, genes, fmr_data_path + f'CAP_{DATE}_{key}_all.csv')    # 바뀜
    for comp, EXPd in COMPS.items():
        if comp == 'Comp3' and t == 'ko':
            continue                                   # 예전 코드처럼 TrainOnly 는 WT 만
        e_c, b_c, c_c = EXPd[key]
        caps = {'all': cap_all}
        for lab in sorted(c_c.unique()):
            idx = c_c.index[c_c == lab]
            if len(idx) < min_n:
                print(f'skip {key} {comp} {lab}: n={len(idx)}')
                continue
            caps[lab] = build_cap(e_c, idx, genes, fmr_data_path + f'CAP_{DATE}_{key}_{lab}.csv')   # 바뀜
        CAPS[(key, comp)] = caps

















from scipy.stats import spearmanr, hypergeom
import matplotlib.pyplot as plt

def pair(comp, t):
    return {'Comp1': ('yo', 'tr'),
            'Comp2': (f'{t}_PM', f'{t}_GM'),
            'Comp3': ('PM', 'GM')}[comp]


def friend_change(caps, base, target):
    """유전자마다 1 - corr(base CAP 행, target CAP 행). 클수록 같이 묶이는 친구가 많이 바뀜"""
    c0, c1 = caps[base], caps[target]
    return pd.Series({g: 1 - np.corrcoef(c0.loc[g].drop(g), c1.loc[g].drop(g))[0, 1]
                      for g in c0.index})

def compare_wt_ko(s, comp='Comp1', top_frac=0.1):
    tabs = {}
    for t in ['wt', 'ko']:
        key = f'{t}_{s}'
        b, tg = pair(comp, t)
        sh = SHIFT[(key, comp)]['shift']
        tabs[t] = pd.DataFrame({'norm': sh['norm'],
                                'friend_change': friend_change(CAPS[(key, comp)], b, tg)})
    genes = tabs['wt'].index.intersection(tabs['ko'].index)
    df = pd.concat([tabs[t].loc[genes].add_prefix(f'{t}_') for t in ['wt', 'ko']], axis=1)
    k = int(top_frac * len(genes))
    print(f'== {s} {comp}: genes={len(genes)}, top {int(top_frac*100)}% = {k} ==')
    for m in ['norm', 'friend_change']:
        rho = spearmanr(df[f'wt_{m}'], df[f'ko_{m}'], nan_policy='omit')[0]
        top_wt = set(df[f'wt_{m}'].nlargest(k).index)
        top_ko = set(df[f'ko_{m}'].nlargest(k).index)
        ov = len(top_wt & top_ko)
        p = hypergeom.sf(ov - 1, len(genes), k, k)
        print(f'  {m}: rank rho={rho:.3f}, top overlap={ov}/{k} (expected {k*k/len(genes):.1f}, p={p:.2e})')
        df[f'wt_{m}_pct'] = df[f'wt_{m}'].rank(pct=True)
        df[f'ko_{m}_pct'] = df[f'ko_{m}'].rank(pct=True)
        df[f'{m}_wt_minus_ko'] = df[f'wt_{m}_pct'] - df[f'ko_{m}_pct']   # + 면 WT 에서만 많이 바뀜
    return df

def plot_wt_ko(df, m='friend_change', title='', highlight=None, savepath=None):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(df[f'wt_{m}_pct'], df[f'ko_{m}_pct'], s=8, alpha=0.4, color='gray', linewidth=0)
    for g in (highlight or []):
        if g in df.index:
            x, y = df.loc[g, [f'wt_{m}_pct', f'ko_{m}_pct']]
            ax.scatter(x, y, s=30, color='orangered')
            ax.text(x, y, g, fontsize=7)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.6)
    ax.set_xlabel(f'WT {m} (percentile)'); ax.set_ylabel(f'KO {m} (percentile)')
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    return fig





import os
import numpy as np
import pandas as pd
import cscore as cscr

DATE = '20261106'                      # CAP 저장할 때 쓴 날짜
groups = [(t, s) for t in ['wt', 'ko'] for s in ['UB', 'LB']]

def pair(comp, t):
    return {'Comp1': ('yo', 'tr'),
            'Comp2': (f'{t}_PM', f'{t}_GM'),
            'Comp3': ('PM', 'GM')}[comp]

def load_cap(data_path, key, lab, genes=None):
    path = fmr_data_path + f'CAP_{DATE}_{key}_{lab}.csv'
    if not os.path.exists(path):
        print('없음:', path)
        return None
    cap = pd.read_csv(path, index_col=0)
    genes = list(cap.index) if genes is None else genes
    cap = cap.reindex(index=genes, columns=genes).fillna(0.0)
    a = cap.values.copy(); np.fill_diagonal(a, 1)
    return pd.DataFrame(a, index=genes, columns=genes)

# CAPS 다시 만들기 (파일에서 읽기만)
CAPS = {}
for t, s in groups:
    key = f'{t}_{s}'
    cap_all = load_cap(fmr_data_path, key, 'all')
    genes = list(cap_all.index)
    for comp in ['Comp1', 'Comp2', 'Comp3']:
        if comp == 'Comp3' and t == 'ko':
            continue
        labs = list(pair(comp, t)) + (['yo'] if comp == 'Comp3' else [])
        caps = {'all': cap_all}
        for lab in labs:
            c = load_cap(key, lab, genes)
            if c is not None:
                caps[lab] = c
        CAPS[(key, comp)] = caps

# SHIFT 다시 계산 (금방 끝남)
SHIFT = {}
for (key, comp), caps in CAPS.items():
    base, target = pair(comp, key.split('_')[0])
    if base in caps and target in caps:
        SHIFT[(key, comp)] = cscr.cap_shift(caps, base=base, target=target)

print(list(SHIFT.keys()))






fmr_plot_dir = '/Users/jiyeonhan/Dropbox/NYU_LIFE/Memory_ML/02.FMR1_Project/FMR1_Project/figure/'

# =====================================================================
# 0. 공통 설정
# =====================================================================
from scipy.stats import wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt

COMPS_ALL  = {'Comp1': 'tr vs yo', 'Comp2': 'GM vs PM', 'Comp3': 'trGM vs trPM'}
COMPS_WTKO = ['Comp1', 'Comp2']                   # KO 도 있는 Comp (Comp3 는 WT 만)
EXPD = {'Comp1': EXP_yotr, 'Comp2': EXP_pmgm, 'Comp3': EXP_yopg}

def cond_caps(t, s, comp):
    """(base CAP, target CAP)"""
    caps = CAPS[(f'{t}_{s}', comp)]
    b, g = pair(comp, t)
    return caps[b], caps[g]

def cond_samples(t, s, comp):
    """base + target 조건에 해당하는 샘플만"""
    e, _, c = EXPD[comp][f'{t}_{s}']
    idx = c.index[c.isin(pair(comp, t))]
    return e.loc[idx], c.loc[idx]

def module_cap_density(cap, partition, n_perm=1000, seed=0):
    """모듈 안 유전자 쌍의 평균 CAP. null = 같은 크기의 무작위 유전자 묶음"""
    genes = [g for g in partition if g in cap.index]
    lab = pd.Series(partition).loc[genes].values
    A = cap.loc[genes, genes].values
    rng = np.random.default_rng(seed)
    rows = []
    for m in sorted(set(lab)):
        idx = np.where(lab == m)[0]; n = len(idx)
        if n < 3:
            continue
        iu = np.triu_indices(n, 1)
        obs = A[np.ix_(idx, idx)][iu].mean()
        null = np.array([A[np.ix_(r, r)][iu].mean()
                         for r in (rng.choice(len(genes), n, replace=False) for _ in range(n_perm))])
        rows.append(dict(module=f'module_{m}', n_genes=n, density=obs,
                         z=(obs - null.mean()) / null.std(), p=(1 + (null >= obs).sum()) / (n_perm + 1)))
    return pd.DataFrame(rows).set_index('module')


# =====================================================================
# 1. 유전자 단위 WT vs KO (Comp1, Comp2)  -- 기존 코드 그대로
# =====================================================================
WK = {}
for s in ['UB', 'LB']:
    WK[s] = {}
    for comp in COMPS_WTKO:
        df = compare_wt_ko(s, comp)
        df.to_csv(fmr_data_path + f'wt_vs_ko_{DATE}_{s}_{comp}.csv')
        plot_wt_ko(df, 'friend_change', f'{s} {comp} ({COMPS_ALL[comp]})', required_gene_list,
                   savepath=fmr_plot_dir + f'wt_vs_ko_{s}_{comp}.jpg')
        WK[s][comp] = df


# =====================================================================
# 2. 모듈 k 스캔: (유전형, sub, Comp) 마다 base + target CAP concat
# =====================================================================
MODS = {}
for comp in COMPS_ALL:
    for t in ['wt', 'ko']:
        for s in ['UB', 'LB']:
            if (f'{t}_{s}', comp) not in CAPS:
                continue                                          # Comp3 KO 없음
            cb, ct = cond_caps(t, s, comp)
            title = f'{t.upper()} {s} {comp} ({COMPS_ALL[comp]})'
            scan = cscr.scan_caps([cb, ct], min_modules=3, max_modules=15, min_size=10, title=title)
            cscr.plot_find_modules(scan['results_df'], title,
                                   savepath=fmr_plot_dir + f'k_scan_{t}_{s}_{comp}.jpg')
            e, c = cond_samples(t, s, comp)
            genes = list(scan['sim'].index)
            ktab = cscr.refilter_k_table(cscr.k_table(scan, e[genes]),
                                         cscr.redundancy_by_k(scan, e[genes], groups=c))
            print(f'\n== {title} ==\n', ktab[['silhouette', 'min_size', 'stab_ari', 'n_pos_redundant_w']])
            MODS[(t, s, comp)] = dict(scan=scan, ktab=ktab)


# =====================================================================
# 3. k 정하기 -> partition + clustermap   (표 보고 채우기)
# =====================================================================
K = {('wt', 'UB', 'Comp1'): 9, ('wt', 'LB', 'Comp1'): 5,
     ('ko', 'UB', 'Comp1'): 3, ('ko', 'LB', 'Comp1'): 6,
     ('wt', 'UB', 'Comp2'): 3, ('wt', 'LB', 'Comp2'): 3, 
     ('ko', 'UB', 'Comp2'): 12, ('ko', 'LB', 'Comp2'): 5,
     ('wt', 'UB', 'Comp3'): 3, ('wt', 'LB', 'Comp3'): 3,
     }


for (t, s, comp), k in K.items():
    m = MODS[(t, s, comp)]
    m['partition'] = cscr.partition_from_scan(m['scan'], k)
    cscr.plot_cap_clustermap_real(f'{t.upper()} {s} {comp} (k={k})', m['scan']['sim'], m['partition'],
                                  savepath=fmr_plot_dir + f'clustermap_{t}_{s}_{comp}.jpg')


# =====================================================================
# 4. WT 모듈이 KO 에서 어떻게 되나 (Comp1, Comp2)
# =====================================================================
def plot_cap_grid(s, comp, savepath=None):
    part = MODS[('wt', s, comp)]['partition']
    order = sorted(part, key=lambda g: (part[g], g))
    bounds = np.cumsum(pd.Series(part).value_counts().sort_index().values)[:-1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, t in enumerate(['wt', 'ko']):
        for j, (cap, lab) in enumerate(zip(cond_caps(t, s, comp), pair(comp, t))):
            ax = axes[i, j]
            sns.heatmap(cap.loc[order, order], ax=ax, cmap='Blues', vmin=0, vmax=1, cbar=False,
                        xticklabels=False, yticklabels=False, rasterized=True)
            for b in bounds:
                ax.axhline(b, color='red', lw=0.5); ax.axvline(b, color='red', lw=0.5)
            ax.set_title(f'{t.upper()} {lab}')
    fig.suptitle(f'{s} {comp}: CAP ordered by WT modules')
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)




SUMMARY = {}
for comp in COMPS_WTKO:
    for s in ['UB', 'LB']:
        if ('wt', s, comp) not in K or ('ko', s, comp) not in K:
            continue
        pw, pk = MODS[('wt', s, comp)]['partition'], MODS[('ko', s, comp)]['partition']
        caps4 = dict(zip(['wt_base', 'wt_target'], cond_caps('wt', s, comp)))
        caps4.update(zip(['ko_base', 'ko_target'], cond_caps('ko', s, comp)))
        # (a) WT 모듈이 각 조건 CAP 에서 얼마나 뭉쳐 있나
        dens = pd.concat({n: module_cap_density(c, pw)[['density', 'z']] for n, c in caps4.items()}, axis=1)
        dens.columns = [f'{a}_{b}' for a, b in dens.columns]
        # (b) KO 자체 모듈에서도 WT 모듈 유전자 쌍이 같이 묶이나
        pres = cscr.compute_pairwise_preservation(pw, pk).set_index('module')[['preserve_rate']]
        cscr.compare_partitions(pw, pk, name=f'{s} {comp} WT vs KO')
        cscr.plot_partition_crosstab(f'{s} {comp}', pw, pk, a_label='WT', b_label='KO',
                                     savepath=fmr_plot_dir + f'wt_ko_crosstab_{s}_{comp}')
        cscr.plot_cluster_sankey(pd.Series(pw), pd.Series(pk), label_base='WT', label_target='KO',
                                 savepath=fmr_plot_dir + f'wt_ko_sankey_{s}_{comp}.html')
        # (c) 모듈별 friend_change
        fc = WK[s][comp][['wt_friend_change', 'ko_friend_change']]
        fc = fc.loc[fc.index.intersection(list(pw))].assign(module=lambda d: [f'module_{pw[g]}' for g in d.index])
        resp = fc.groupby('module').agg(wt_fc=('wt_friend_change', 'mean'), ko_fc=('ko_friend_change', 'mean'))
        resp['p_wilcoxon'] = [wilcoxon(g['wt_friend_change'], g['ko_friend_change'])[1] if len(g) >= 5 else np.nan
                              for _, g in fc.groupby('module')]
        summary = dens.join(pres, how='left').join(resp, how='left')
        summary.to_csv(fmr_data_path + f'module_preservation_{DATE}_{s}_{comp}.csv')
        SUMMARY[(s, comp)] = summary
        plot_cap_grid(s, comp, savepath=fmr_plot_dir + f'cap_grid_{s}_{comp}.jpg')
        print(f'\n== {s} {comp} ==\n', summary.round(3))


# =====================================================================
# 5. Comp 끼리 비교: 같은 유전형, 같은 sub 에서 모듈 구성이 얼마나 비슷한가
# =====================================================================
rows = []
for t in ['wt', 'ko']:
    for s in ['UB', 'LB']:
        comps = [c for c in COMPS_ALL if (t, s, c) in K]
        for i, a in enumerate(comps):
            for b in comps[i + 1:]:
                ari, nmi, n = cscr.ari_between(MODS[(t, s, a)]['partition'], MODS[(t, s, b)]['partition'])
                rows.append(dict(genotype=t, sub=s, comp_a=a, comp_b=b, ari=ari, nmi=nmi, n_genes=n))
COMP_CMP = pd.DataFrame(rows)
COMP_CMP.to_csv(fmr_data_path + f'comp_partition_similarity_{DATE}.csv', index=False)
print(COMP_CMP.round(3))










from scipy.stats import rankdata, spearmanr

def cap_dict(s, comp):
    """{'WT yo': CAP, 'WT tr': CAP, 'KO yo': ..., 'KO tr': ...}"""
    d = {}
    for t in ['wt', 'ko']:
        if (f'{t}_{s}', comp) not in CAPS:
            continue
        for cap, lab in zip(cond_caps(t, s, comp), pair(comp, t)):
            d[f'{t.upper()} {lab.split("_")[-1]}'] = cap
    return d

def delta_compare(s, comp):
    wb, wt_ = cond_caps('wt', s, comp)
    kb, kt = cond_caps('ko', s, comp)
    genes = wb.index.intersection(kb.index)
    dW = (wt_ - wb).loc[genes, genes].values
    dK = (kt - kb).loc[genes, genes].values
    iu = np.triu_indices(len(genes), 1)
    rho_all = spearmanr(dW[iu], dK[iu])[0]
    rW, rK = rankdata(dW, axis=1), rankdata(dK, axis=1)
    rW = (rW - rW.mean(1, keepdims=True)) / rW.std(1, keepdims=True)
    rK = (rK - rK.mean(1, keepdims=True)) / rK.std(1, keepdims=True)
    df = pd.DataFrame({'concordance': (rW * rK).mean(1),          # 재배선 패턴이 WT, KO 에서 같은 정도
                       'wt_change': np.abs(dW).mean(1),             # 재배선 크기
                       'ko_change': np.abs(dK).mean(1)}, index=genes)
    return rho_all, df

EDGE, DELTA = {}, {}
for comp in COMPS_ALL:
    for s in ['UB', 'LB']:
        d = cap_dict(s, comp)
        tab = cscr.cap_edge_table(d)
        EDGE[(s, comp)] = tab
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(tab, annot=True, fmt='.2f', cmap='viridis', vmin=0, vmax=1, ax=ax)
        ax.set_title(f'{s} {comp}: CAP edge Spearman', fontsize=9)
        fig.savefig(fmr_plot_dir + f'cap_edge_sim_{s}_{comp}.jpg', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'\n== {s} {comp} ==\n', tab.round(3))
        if comp in COMPS_WTKO:
            rho, df = delta_compare(s, comp)
            df.to_csv(fmr_data_path + f'delta_concordance_{DATE}_{s}_{comp}.csv')
            DELTA[(s, comp)] = df
            print(f'  dCAP WT vs KO (all edges): rho={rho:.3f}')
            print('  WT, KO 재배선 같은 쪽:', df.sort_values('concordance', ascending=False).head(10).index.tolist())
            print('  반대 쪽:', df.sort_values('concordance').head(10).index.tolist())









from scipy.cluster.hierarchy import linkage, fcluster

def friend_change(caps, base, target):
    c0, c1 = caps[base], caps[target]
    return pd.Series({g: 1 - np.corrcoef(c0.loc[g].drop(g), c1.loc[g].drop(g))[0, 1] for g in c0.index})

def change_matrix(s):
    """유전자 x 비교 (WT Comp1, KO Comp1, WT Comp2, ...) friend_change 백분위"""
    cols = {}
    for comp in COMPS_ALL:
        for t in ['wt', 'ko']:
            if (f'{t}_{s}', comp) not in CAPS:
                continue
            b, g = pair(comp, t)
            cols[f'{t.upper()} {comp}'] = friend_change(CAPS[(f'{t}_{s}', comp)], b, g)
    M = pd.DataFrame(cols).dropna()
    return M.rank(pct=True)




def rule_groups(M, a='WT Comp1', b='KO Comp1', hi=0.75, lo=0.5):
    g = pd.Series('other', index=M.index)
    g[(M[a] >= hi) & (M[b] <  lo)] = 'WT only'      # WT 에서만 크게 바뀜
    g[(M[a] <  lo) & (M[b] >= hi)] = 'KO only'      # KO 에서만 크게 바뀜
    g[(M[a] >= hi) & (M[b] >= hi)] = 'both'         # 둘 다 크게 바뀜
    g[(M[a] <  lo) & (M[b] <  lo)] = 'stable'       # 둘 다 작게 바뀜
    return g


PAT = {}
for s in ['UB', 'LB']:
    M = change_matrix(s)
    for comp in COMPS_WTKO:                                   # Comp1, Comp2 각각
        lab = rule_groups(M, a=f'WT {comp}', b=f'KO {comp}')
        part = lab[lab != 'other'].to_dict()
        # (b) 그룹별 평균 변화 + candidate
        prof = M.loc[list(part)].groupby(pd.Series(part)).mean().round(2)
        prof['n_genes'] = pd.Series(part).value_counts()
        prof['candidates'] = [', '.join(g for g in required_gene_list if part.get(g) == p) for p in prof.index]
        # (c) 그룹이 각 조건 CAP 에서 얼마나 뭉쳐 있나 (density z)
        dens = {}
        for c2 in COMPS_ALL:
            for t in ['wt', 'ko']:
                if (f'{t}_{s}', c2) not in CAPS:
                    continue
                for cap, c in zip(cond_caps(t, s, c2), pair(comp=c2, t=t)):
                    short = c.split('_')[-1]
                    name = f'{t.upper()} {short}' if short in ('yo', 'tr') else f'{t.upper()} {short} ({c2})'
                    if name not in dens:                      # yo 만 Comp1, Comp3 에 중복
                        dens[name] = module_cap_density(cap, part)['z']
        dens = pd.DataFrame(dens)
        dens.index = [i.replace('module_', '') for i in dens.index]
        prof.to_csv(fmr_data_path + f'change_group_profile_{DATE}_{s}_{comp}.csv')
        dens.to_csv(fmr_data_path + f'change_group_density_{DATE}_{s}_{comp}.csv')
        PAT[(s, comp)] = dict(M=M, label=lab, profile=prof, density=dens)
        print(f'\n== {s} {comp}: 그룹별 변화 (백분위 평균) ==\n', prof)
        print(f'\n== {s} {comp}: 그룹별 CAP 안 뭉침 정도 (z) ==\n', dens.round(1))








from scipy.stats import spearmanr

G = PAT[('LB', 'Comp2')]['label']
G = list(G.index[G == 'both'])
print('genes:', len(G))

# --- 샘플 그룹 (발현 기준) ---
def samples(t, s, col, val):
    e, _, c = {'training': EXP_yotr, 'new_tr_tr': EXP_pmgm}[col][f'{t}_{s}']
    return e.loc[c.index[c == val]]

GRP = {'WT yo': samples('wt', 'LB', 'training', 'yo'),  'WT tr': samples('wt', 'LB', 'training', 'tr'),
       'WT GM': samples('wt', 'LB', 'new_tr_tr', 'wt_GM'),
       'KO yo': samples('ko', 'LB', 'training', 'yo'),  'KO tr': samples('ko', 'LB', 'training', 'tr'),
       'KO GM': samples('ko', 'LB', 'new_tr_tr', 'ko_GM')}

# (2) 발현 기준: 그룹 유전자끼리 |Spearman| 평균, 같은 샘플에서 무작위 유전자 묶음과 비교한 z
def expr_coherence(e, genes, n_perm=500, seed=0):
    genes = [g for g in genes if g in e.columns]
    def mean_abs(gs):
        r = spearmanr(e[gs].values).correlation
        return np.nanmean(np.abs(r[np.triu_indices(len(gs), 1)]))
    obs = mean_abs(genes)
    rng = np.random.default_rng(seed)
    pool = list(e.columns[e.std() > 0])
    null = np.array([mean_abs(list(rng.choice(pool, len(genes), replace=False))) for _ in range(n_perm)])
    return dict(n=len(e), mean_abs_r=obs, z=(obs - null.mean()) / null.std())

EXPR_CHECK = pd.DataFrame({k: expr_coherence(v, G) for k, v in GRP.items()}).T
print(EXPR_CHECK.round(3))

# (3) CAP 기준: 이 유전자들 사이 CAP 구조가 조건끼리 얼마나 닮았나
CAPG = {'WT yo': CAPS[('wt_LB', 'Comp1')]['yo'], 'WT tr': CAPS[('wt_LB', 'Comp1')]['tr'],
        'WT GM': CAPS[('wt_LB', 'Comp2')]['wt_GM'],
        'KO yo': CAPS[('ko_LB', 'Comp1')]['yo'], 'KO tr': CAPS[('ko_LB', 'Comp1')]['tr'],
        'KO GM': CAPS[('ko_LB', 'Comp2')]['ko_GM']}
iu = np.triu_indices(len(G), 1)
V = {k: c.loc[G, G].values[iu] for k, c in CAPG.items()}
SIM = pd.DataFrame({a: {b: spearmanr(V[a], V[b])[0] for b in V} for a in V})
print(SIM.round(3))





from gprofiler import GProfiler
gp = GProfiler(return_dataframe=True)

GS = {}
for s in ['UB', 'LB']:
    w = pd.read_csv(fmr_data_path + f'gene_scores_wt_{s}.csv', index_col=0).set_index('ID')
    k = pd.read_csv(fmr_data_path + f'gene_scores_ko_{s}.csv', index_col=0).set_index('ID')
    df = w[['SCOR', 'Z_pv']].add_prefix('wt_').join(k[['SCOR', 'Z_pv']].add_prefix('ko_'), how='inner')
    ko_only = df[(df.ko_Z_pv < 0.05) & (df.wt_Z_pv > 0.5)]     # KO 에서만 행동과 관련
    both    = df[(df.ko_Z_pv < 0.05) & (df.wt_Z_pv < 0.05)]    # 둘 다
    print(f'{s}: KO only {len(ko_only)}, both {len(both)} '
          f'(both 중 방향 같음 {int((np.sign(both.wt_SCOR) == np.sign(both.ko_SCOR)).sum())})')
    go = gp.profile(organism='mmusculus', query=list(ko_only.index), background=list(df.index),
                    domain_scope='custom', sources=['GO:BP', 'GO:CC', 'GO:MF', 'KEGG', 'REAC'],
                    user_threshold=0.05, significance_threshold_method='fdr', no_evidences=True)
    if go is not None and len(go):
        go = go[(go.term_size >= 10) & (go.term_size <= 500) & (go.intersection_size >= 3)]
    df.to_csv(fmr_data_path + f'behavior_genes_wt_vs_ko_{DATE}_{s}.csv')
    GS[s] = dict(table=df, ko_only=ko_only, both=both, go=go)
    print(go[['source', 'name', 'p_value', 'intersection_size']].head(15) if go is not None and len(go) else '  GO term 없음')








from scipy.stats import rankdata

def sigma_strength(t, s, comp, lab, genes):
    """그 조건 샘플로 pairwise SIGMA 를 다시 계산해서 유전자별 합"""
    e, _, c = EXPD[comp][f'{t}_{s}']
    idx = c.index[c == lab]
    long = cscr.pairwise_single(e.loc[idx], list(genes))
    st = long.groupby('geneA')['SIGMA'].sum().add(long.groupby('geneB')['SIGMA'].sum(), fill_value=0)
    return st.reindex(genes).fillna(0)

def row_z(A):
    R = np.apply_along_axis(rankdata, 1, A)
    return (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True)



def hub_substitution(s, comp='Comp1', top=0.1, n_perm=200, seed=0):
    lw, lk = pair(comp, 'wt')[1], pair(comp, 'ko')[1]                 # target 조건 (tr 또는 GM)
    cw, ck = CAPS[(f'wt_{s}', comp)][lw], CAPS[(f'ko_{s}', comp)][lk]
    genes = cw.index.intersection(ck.index)
    cw, ck = cw.loc[genes, genes], ck.loc[genes, genes]
    # 허브: SIGMA 네트워크 strength (조건 안 백분위)
    sw = sigma_strength('wt', s, comp, lw, genes).rank(pct=True)
    sk = sigma_strength('ko', s, comp, lk, genes).rank(pct=True)
    # 1. 허브 겹침
    hw, hk = set(sw.index[sw >= 1 - top]), set(sk.index[sk >= 1 - top])
    n = len(hw)
    print(f'{s} {comp} ({lw} vs {lk}): hub {n}, 겹침 {len(hw & hk)} (우연 기대 {n * n / len(genes):.1f})')
    # 2. 잃어버린 허브 (WT 허브인데 KO 에서 하위 50%)
    lost = [g for g in hw if sk[g] < 0.5]
    if not lost:
        print('  잃어버린 허브 없음')
        return pd.DataFrame(), dict(wt=sw, ko=sk)
    # 3. 대체 후보: WT 에서 그 허브의 CAP 친구 목록과 가장 비슷한 KO 유전자
    A, B = cw.values.copy().astype(float), ck.values.copy().astype(float)
    np.fill_diagonal(A, np.nan); np.fill_diagonal(B, np.nan)
    A = np.where(np.isnan(A), np.nanmean(A, 1, keepdims=True), A)
    B = np.where(np.isnan(B), np.nanmean(B, 1, keepdims=True), B)
    ZA, ZB = row_z(A), row_z(B)
    pos = {g: i for i, g in enumerate(genes)}
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(len(genes)) for _ in range(n_perm)]
    rows = []
    for h in lost:
        i = pos[h]
        r = ZB @ ZA[i] / len(genes)                                     # WT 의 h 행 vs KO 모든 행
        j = int(np.argmax(np.where(np.arange(len(genes)) == i, -np.inf, r)))
        null = np.array([np.max(ZB[p][:, p] @ ZA[i] / len(genes)) for p in perms])
        rows.append(dict(hub=h, wt_pct=sw[h], ko_pct=sk[h], self_corr=r[i],
                         substitute=genes[j], sub_corr=r[j], sub_ko_pct=sk[genes[j]],
                         p_null=(1 + np.sum(null >= r[j])) / (n_perm + 1)))
    out = pd.DataFrame(rows).sort_values('p_null')
    return out, dict(wt=sw, ko=sk)


SUB = {}
for comp in COMPS_WTKO:
    for s in ['UB', 'LB']:
        tab, st = hub_substitution(s, comp)
        if len(tab):
            tab.to_csv(fmr_data_path + f'hub_substitution_{DATE}_{s}_{comp}.csv', index=False)
            print(tab.head(10).round(3))
            print('  substitute 빈도:', tab['substitute'].value_counts().head(5).to_dict())
        SUB[(s, comp)] = tab









from scipy.stats import spearmanr, hypergeom

# --- 1 day 유전자 점수: 저장된 파일 읽기 ---
res_d1 = pd.read_csv(out_dir + 'gene_scores_day1.csv', index_col=0).set_index('ID')

def load_cap_file(path):
    cap = pd.read_csv(path, index_col=0)
    a = cap.values.copy().astype(float); np.fill_diagonal(a, 1)
    return pd.DataFrame(a, index=cap.index, columns=cap.columns)

D1 = {'yo': load_cap_file(out_dir + 'CAP_0.csv'), 'tr': load_cap_file(out_dir + 'CAP_1.csv')}
fc_d1 = friend_change(D1, 'yo', 'tr').rank(pct=True)

def overlap(a, b, universe):
    a, b = set(a) & universe, set(b) & universe
    x = len(a & b)
    return x, len(a) * len(b) / len(universe), hypergeom.sf(x - 1, len(universe), len(a), len(b))

# --- 비교 ---
rows = []
for s in ['UB', 'LB']:
    # (1) friend_change 순위: 두 데이터셋 CAP 에 공통으로 있는 유전자만
    M = PAT[(s, 'Comp1')]['M']
    genes = M.index.intersection(fc_d1.index)
    k = max(int(0.1 * len(genes)), 1)
    top_d1 = set(fc_d1.loc[genes].nlargest(k).index)
    for t in ['WT', 'KO']:
        wk = M.loc[genes, f'{t} Comp1']
        x, e, p = overlap(top_d1, wk.nlargest(k).index, set(genes))
        rows.append(dict(sub=s, compare=f'1day WT vs 1week {t}', measure='friend_change',
                         n_genes=len(genes), rho=spearmanr(fc_d1.loc[genes], wk)[0],
                         overlap=x, expected=round(e, 1), p=p))
    # (2) 행동 관련 유전자 (전체 유전자 기준)
    for t in ['wt', 'ko']:
        gw = pd.read_csv(fmr_data_path + f'gene_scores_{t}_{s}.csv', index_col=0).set_index('ID')
        univ = set(gw.index) & set(res_d1.index)
        x, e, p = overlap(res_d1.index[res_d1.Z_pv < 0.05], gw.index[gw.Z_pv < 0.05], univ)
        rows.append(dict(sub=s, compare=f'1day WT vs 1week {t.upper()}', measure='behavior genes',
                         n_genes=len(univ), rho=np.nan, overlap=x, expected=round(e, 1), p=p))

D1CMP = pd.DataFrame(rows)
D1CMP.to_csv(fmr_data_path + f'day1_vs_week1_{DATE}.csv', index=False)
print(D1CMP.round(3))









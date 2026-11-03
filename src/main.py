



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.stats import norm
from joblib import Parallel, delayed
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgba, to_rgb
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import networkx as nx
import community
from community import community_louvain
from gprofiler import GProfiler
from collections import defaultdict
from pyvis.network import Network
from sklearn.decomposition import PCA
from tqdm import tqdm
import copy
import colorsys
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import rcParams
import os 

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import markov_clustering as mc
import plotly.graph_objects as go
import kaleido

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter


datapath = './data/'
plotpath = './figures/'

# for pdf saving, text to vector 
rcParams['pdf.fonttype'] = 42 
rcParams['ps.fonttype'] = 42   # EPS 

# Arial 
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']


# notations for new methods in figures
s_xi = r'$\xi_c$'
s_xicor = r'$\tilde\xi_c$'
s_pcc = r'$r_p$'
s_scc = r'$\rho_s$'
s_Sigma = r'$\Sigma$'
s_Sigma_t = r'$\tilde{\Sigma}$'




# 아예 input 을 나눠서 받게 하는게 나을것 같음 



for cond, sub in product(['wt', 'ko'], ['LB', 'UB']):
    print(f"Running PC Analysis for {cond} {sub}...")
    r_f = filter_data_sub_treat(RNA_TOT, sub, cond)
    p_f = filter_data_sub_treat(PROT_TOT, sub, cond)
    _ = PC_based_parallel_do(f'{DATE}.RNA.{cond}.{sub}', r_f, RNA_TOT_genes, 'PC1', RNA_gene_ID)
    _ = PC_based_parallel_do(f'{DATE}.PROT.{cond}.{sub}', p_f, PROT_TOT_genes, 'PC1', Prot_gene_ID)


for cond, sub in product(['wt', 'ko'], ['LB', 'UB']):
    df = pd.read_csv(processed_dir+f'03.Pall_PC_{DATE}.RNA.{cond}.{sub}.csv', index_col = 0)
    draw_double_scatter(df, f'{DATE}.RNA.{cond}.{sub}', RNA_candy)
    df = pd.read_csv(processed_dir+f'03.Pall_PC_{DATE}.PROT.{cond}.{sub}.csv', index_col = 0)
    draw_double_scatter(df, f'{DATE}.PROT.{cond}.{sub}', Prot_candy)


p_f = filter_data_sub_treat(PROT_TOT, 'LB', 'ko')
small_pc_gene_scatter('Fmr1', Prot_gene_ID, p_f)
    

def check_union_genes():
    gene_dict = {}
    for sub in ['LB', 'UB']:
        R_union = get_selected_genes('RNA', sub)
        P_union = get_selected_genes('PROT', sub)
        add_candy_R_union = list(np.unique(R_union + list(RNA_candy.ID)))
        add_candy_P_union = list(np.unique(P_union + list(Prot_candy.ID)))
        gene_dict[f'{sub}_RNA'] = add_candy_R_union
        gene_dict[f'{sub}_PROT'] = add_candy_P_union
        print(f"{sub}_RNA before : {len(R_union)} -> after  : {len(add_candy_R_union)}")
        print(f"{sub}_PROT before : {len(P_union)} -> after  : {len(add_candy_P_union)}")
    return gene_dict


this_gene_dict = check_union_genes()

condi_rules = {
    "yo":  {"label": "YOKED", "func": lambda df: df[df.tr_tr.isin(['wt_yo','ko_yo'])]},
    "tr":  {"label": "TRAINED", "func": lambda df: df[df.tr_tr.isin(['wt_tr','ko_tr'])]},
    "PM":  {"label": "PM", "func": lambda df: df[df.new_tr_tr.isin(['wt_PM','ko_PM'])]},
    "GM":  {"label": "GM", "func": lambda df: df[df.new_tr_tr.isin(['wt_GM','ko_GM'])]},
    "TrainOnly.PM": {"label": "TrainOnly.PM",
                    "func": lambda df: df[df.tr_tr.isin(['wt_tr','ko_tr']) & df.new_tr_tr.isin(['wt_PM','ko_PM'])]},
    "TrainOnly.GM": {"label": "TrainOnly.GM",
                    "func": lambda df: df[df.tr_tr.isin(['wt_tr','ko_tr']) & df.new_tr_tr.isin(['wt_GM','ko_GM'])]},
}



iteration = 1000
for sub in ['LB', 'UB']:
    for cond in ['wt', 'ko']:
        R_union = this_gene_dict[f'{sub}_RNA']
        P_union = this_gene_dict[f'{sub}_PROT']
        run_pairwise_only('SINGLE', cond, sub, R_union, P_union)
        run_pairwise_only('RNA_PROT', cond, sub, R_union, P_union)
        run_clustering_only('SINGLE', cond, sub, iteration)
        run_clustering_only('RNA_PROT', cond, sub, iteration)




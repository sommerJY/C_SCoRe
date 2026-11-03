
def small_pc_gene_scatter(name, dict_name, df) :
    name_id = dict_name[dict_name['gene'] == name]['ID'].item()
    fig,axes = plt.subplots(1, 1, figsize=(3, 3))
    sns.scatterplot(x='PC1', y=name_id, data=df, ax=axes)
    axes.set_xlabel('PC1')
    axes.set_ylabel(name)
    plt.tight_layout()
    plt.show()


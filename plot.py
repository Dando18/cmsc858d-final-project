import os.path


def plot_reduced_data(Z, dataset, fname, title):
    ''' Plot 2d data from dataset into fname
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt

    OUTPUT_DIRECTORY = 'figures'
    OUTPUT_PATH = os.path.join(OUTPUT_DIRECTORY, fname)

    plt.clf()
    plt.title(title)
    sns.scatterplot(x=Z[:,0], y=Z[:,1], hue=dataset['clusterColors'][dataset['clusters']], s=1)
    plt.legend([],[], frameon=False)
    plt.savefig(OUTPUT_PATH, dpi=600)
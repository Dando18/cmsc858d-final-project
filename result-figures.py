''' Generate figures for the paper.
'''
from itertools import product
import os.path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import pandas as pd


DATASETS = ['mouse-exon', 'ca1-neurons', 'pollen']
ALGORITHMS = ['pca', 'lda', 'tsne', 'umap', 'densne', 'densmap']
PERF_METRICS = ['duration', 'knn', 'knc', 'cpd', 'ari', 'pds', 'cs']
DIST_METRICS = ['euclidean', 'l1', 'cosine', 'hamming']


def metric_plots():
    data_fname = 'results/metrics.csv'
    dataset_name = 'mouse-exon'
    algs = ['tsne', 'umap', 'densmap']
    mets = ['knn', 'knc', 'cpd', 'ari', 'pds', 'cs']

    df = pd.read_csv(data_fname)
    df = df[df['dataset'] == dataset_name]
    df = df[df['metric'] == df['eval_dist_metric']]

    # for each data set create 6 plots 1 for each scoring metric
    # each plot will have a bar for each distance metric

    sns.set()

    for alg in algs:
        sub_df = df[df['algorithm'] == alg]

        plt.clf()
        fig, axs = plt.subplots(3, 2, constrained_layout=True)

        for i, m in enumerate(mets):
            a = sns.barplot(x=sub_df['metric'], y=sub_df[m], ax=axs[i//2,i%2])
            a.bar_label(a.containers[0], fmt='%.2f', fontsize=8)
            a.set_ylim([0,1])
            #a.tick_params(axis='x', rotation=10)

        fig.suptitle('Comparing Distance Metrics\nwith {} on the {} Dataset'.format(alg.capitalize(), dataset_name))
        out_fname = os.path.join('figures', 'metrics_{}_{}_results.png'.format(alg, dataset_name))
        plt.savefig(out_fname, dpi=100)


def alg_plots():
    data_fname = 'results/algorithms.csv'
    datasets = ['mouse-exon', 'ca1-neurons', 'pollen']
    mets = ['knn', 'knc', 'cpd', 'ari', 'pds', 'cs']

    df = pd.read_csv(data_fname)

    # for each data set create 6 plots 1 for each scoring metric
    # each plot will have a bar for each algorithm

    sns.set()

    for dataset in datasets:
        sub_df = df[df['dataset'] == dataset]

        plt.clf()
        fig, axs = plt.subplots(3, 2, constrained_layout=True, figsize=(16,9))

        for i, m in enumerate(mets):
            a = sns.barplot(x=sub_df['algorithm'], y=sub_df[m], ax=axs[i//2,i%2])
            a.bar_label(a.containers[0], fmt='%.2f', fontsize=8)
            a.set_ylim([0,1])
            

        fig.suptitle('Comparing Algorithm Performance\n on the {} Dataset'.format(dataset))
        out_fname = os.path.join('figures', 'algs_{}_results.png'.format(dataset))
        plt.savefig(out_fname, dpi=100)


def runtime_plots():
    data_fname = 'results/algorithms.csv'
    datasets = ['mouse-exon', 'ca1-neurons', 'pollen']
    #mets = ['knn', 'knc', 'cpd', 'ari', 'pds', 'cs']

    df = pd.read_csv(data_fname)

    sns.set()

    for dataset in datasets:
        sub_df = df[df['dataset'] == dataset]
        ylims_by_dataset = {'mouse-exon': [0,50], 'ca1-neurons': [0, 3], 'pollen': [0, 5]}


        plt.clf()
        fig, ax = plt.subplots(1, constrained_layout=True)

        ax = sns.barplot(x='algorithm', y='duration', data=sub_df, ax=ax)
        ax.bar_label(ax.containers[0], fmt='%.2f', label_type='edge', fontsize=8)
        ax.set_ylim(ylims_by_dataset[dataset])
        ax.set_ylabel('Duration (seconds)')

        ax.set_title('Embedding Runtime for each Algorithm\n on the {} Dataset'.format(dataset))
        out_fname = os.path.join('figures', 'runtime_{}_results.png'.format(dataset))
        plt.savefig(out_fname, dpi=100)


def perplexity_plots():
    data_fname = 'results/perplexity.csv'
    datasets = ['mouse-exon', 'ca1-neurons', 'pollen']
    mets = ['knn', 'knc', 'cpd', 'ari', 'pds', 'cs']

    df = pd.read_csv(data_fname)

    sns.set()

    for dataset in datasets:
        sub_df = df[df['dataset'] == dataset]


        plt.clf()
        fig, axs = plt.subplots(3, 2, constrained_layout=True)

        for i, m in enumerate(mets):
            ax = sns.lineplot(x='perplexity', y=m, data=sub_df, ax=axs[i//2,i%2])
            ax.set_xscale('log', base=2)
            ax.set_ylim([0,1])

        fig.suptitle('tSNE Perplexity\n on the {} Dataset'.format(dataset))
        out_fname = os.path.join('figures', 'perplexity_{}_results.png'.format(dataset))
        plt.savefig(out_fname, dpi=100)


def density_plots():
    data_fname = 'results/density.csv'
    datasets = ['mouse-exon', 'ca1-neurons']#, 'pollen']
    mets = ['knn', 'knc', 'cpd', 'ari', 'pds', 'cs']

    df = pd.read_csv(data_fname)

    sns.set()

    for dataset in datasets:
        sub_df = df[df['dataset'] == dataset]
        sub_df = sub_df[sub_df['dens_frac'] == 0.25]

        plt.clf()
        fig, axs = plt.subplots(3, 2, constrained_layout=True)

        for i, m in enumerate(mets):
            ax = sns.lineplot(x='dens_lambda', y=m, hue='algorithm', data=sub_df, ax=axs[i//2,i%2])
            ax.set_xscale('log', base=10)
            ax.set_ylim([0,1])
            if i != 1:
                ax.legend([],[], frameon=False)


        fig.suptitle('Performance Across Density Weight\n on the {} Dataset'.format(dataset))
        out_fname = os.path.join('figures', 'density_{}_results.png'.format(dataset))
        plt.savefig(out_fname, dpi=100)
        


def main():
    metric_plots()
    alg_plots()
    runtime_plots()
    perplexity_plots()
    density_plots()


if __name__ == '__main__':
    main()
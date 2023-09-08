import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.nn import KLDivLoss
import os
import itertools
from scipy import stats
import numpy as np
import seaborn as sns
import difflib
from itertools import chain
from evodiff.utils import extract_seq_a3m, csv_to_dict, normalize_list, removekey, get_matrix, get_pairs, normalize_matrix, \
                      get_pairwise

def aa_reconstruction_parity_plot(project_dir, out_path, generate_file, msa=False, idr=False, gen_file=True,
                                  start_valid=False, start_query=False, start_msa=False):
    "Parity plots for generated vs test (for sequence models) or valid (for MSA models)"
    # Load in approx train distribution
    idr_flag = ""
    # Eliminate BXJOU for KL since they occur at 0 freq in test dataset
    keys_to_remove = ['B', 'Z', 'J', 'O', 'U'] #, '-']
    if msa:
        if start_valid:
            if start_query:
                valid_file = 'valid_msas_onlymsa.txt'
            elif start_msa:
                valid_file = 'valid_msas_onlyquery.txt'
                keys_to_remove += ['-']
            else:
                valid_file = 'valid_msas.a3m'
            valid_file = out_path + '/' + valid_file
            #print(valid_file)
            aminos = csv_to_dict(valid_file)
            values = list(aminos.values())
        else:
            file = project_dir + 'ref/openfold_ref.csv'
    else:
        file = project_dir + 'ref/uniref50_aa_ref_test.csv' # TODO add file to git
        #print(file)
    if idr:
        idr_flag = 'idr_'
        true_file = out_path + 'data_idr.csv'
        aminos = csv_to_dict(true_file)
        values = aminos.values()
        #print(aminos, values)
    elif not idr and not start_valid:
        df = pd.read_csv(file)
        aminos = df.to_dict('list')
        values = [each[0] for each in aminos.values()]
    if gen_file:
        gen_flag = ''
        # Load in generated seqs and count values
        generate_file = out_path + generate_file
        aminos_gen = csv_to_dict(generate_file)
        #print("aminos gen", aminos_gen)
    else:
        gen_flag = '_train_only'
    # Normalize scores
    a = normalize_list(values)  # normalize(list(aminos.values()))
    if start_valid:
        a_kl = normalize_list(list(removekey(aminos, keys_to_remove).values()))
    else:
        #print(aminos)
        a_kl = normalize_list([each[0] for each in removekey(aminos, keys_to_remove).values()])
    if gen_file:
        b_list = list(aminos_gen.values())
        b = normalize_list(b_list) # ADD GAPS IN
        # Save KL to file
        kl_loss = KLDivLoss(reduction="sum")
        if msa:
            b_kl = normalize_list(list(removekey(aminos_gen, keys_to_remove).values()))
            #print(len(a_kl), len(b_kl))
            #print(a_kl, b_kl)
            kl = kl_loss(torch.tensor(a_kl).log(), torch.tensor(b_kl)).item()
        else:
            if idr:
                b_kl = torch.tensor(b[0:20])
                kl = kl_loss(torch.tensor(a[0:20]).log(), torch.tensor(b[0:20])).item()
            else:
                b_kl = torch.tensor(b[0:21])
                kl = kl_loss(torch.tensor(a[0:21]).log(), torch.tensor(b[0:21])).item()
        print("KL", kl)
        with open(out_path + idr_flag + 'generate_metrics.csv', 'w') as f:
            f.write("aa freq kl:" + str(kl))
        f.close()
        kl_label = "$KL$=%.3f" % (kl)

        # Plot
        colors = ['black', 'grey', 'lightcoral', 'brown', 'tomato', 'peru',
                  'darkorange', 'goldenrod', 'khaki', 'olive', 'yellow', 'olivedrab',
                  'yellowgreen', 'palegreen', 'forestgreen', 'turquoise', 'paleturquoise',
                  'cyan', 'deepskyblue', 'dodgerblue', 'royalblue', 'navy', 'blue',
                  'darkslateblue', 'mediumpurple', 'darkviolet', 'violet', 'mediumvioletred',
                  'crimson', 'lightpink']
        fig, ax = plt.subplots(figsize=(3, 2.5))
        annotations = list(aminos_gen.keys())[0:len(a)]
        plt.axline([0, 0], [0.1, 0.1], c='k', linestyle='dotted', alpha=0.75)
        for i, label in enumerate(annotations):
            plt.scatter(a[i], b[i], label=label, c=colors[i], edgecolors='k')
        ax.text(0.05, 0.95, kl_label, transform=ax.transAxes, fontsize=14,
                verticalalignment='top')
        plt.xlabel("Test Freq", fontweight='bold')
        plt.ylabel("Gen Freq", fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(out_path, idr_flag+'parity_scatter.svg'))
        fig.savefig(os.path.join(out_path, idr_flag+'parity_scatter.png'))
        plt.close()
    if not gen_file:
        return a # return train probability distribution


def msa_substitution_rate(generated_msa, train_msa, alphabet, out_path):
    "Plot substitution rates for generated MSAs"
    print(alphabet, "len: ", len(alphabet))
    all_aa = np.arange(len(alphabet))
    all_aa_pairs = list(itertools.product(all_aa, all_aa))

    all_pairs_train = get_pairs(train_msa, alphabet)
    train_matrix = get_matrix(all_pairs_train, all_aa_pairs, alphabet)
    print("train len", len(all_pairs_train))
    train_table, train_vals, train_diag_vals = normalize_matrix(train_matrix.T, alphabet)

    all_pairs_gen = get_pairs(generated_msa, alphabet)
    print("gen len", len(all_pairs_gen))
    gen_matrix = get_matrix(all_pairs_gen, all_aa_pairs, alphabet)
    gen_table, gen_vals, gen_diag_vals = normalize_matrix(gen_matrix.T, alphabet)

    # Plot substitution data as heatmaps
    vmax = 0.4
    fig, ax = plt.subplots(figsize=(3, 2.5))
    sns.heatmap(train_table, annot=False, cmap='Greens', vmin=0, vmax=vmax, ax=ax)
    ax.set_title('Train Substitution Freq', weight='bold', fontsize=14)
    fig.savefig(os.path.join(out_path, 'train_heatmap.svg'))
    fig.savefig(os.path.join(out_path, 'train_heatmap.png'))

    fig, ax = plt.subplots(figsize=(3, 2.5))
    sns.heatmap(gen_table, annot=False, cmap='Greens', vmin=0, vmax=vmax, ax=ax)
    ax.set_title('Gen Substitution Freq', weight='bold', fontsize=14)
    fig.savefig(os.path.join(out_path, 'gen_heatmap.svg'))
    fig.savefig(os.path.join(out_path, 'gen_heatmap.png'))

    # Plot substitution parity per AA
    fig, axes = plt.subplots(6, 5, figsize=(12, 15))
    for i, ax in enumerate(axes.ravel()[:len(alphabet)]):
        r_squared = stats.pearsonr(train_vals[i, :], gen_vals[i, :]).statistic
        label = "$R$=%.2f" % (r_squared)
        # mse = mean_squared_error(train_vals[i,:], gen_vals[i,:])
        # label = "$mse$=%0.2f"%(mse)
        ax.set_title(alphabet[i], fontsize=14, weight='bold')
        ax.plot([0, vmax], [0, vmax], linewidth=1, color='black', linestyle="--")
        ax.scatter(train_vals[i, :], gen_vals[i, :], color='blue',
                   linewidth=0, alpha=1)
        ax.scatter(train_vals[i, i], gen_vals[i, i], color='red',
                   linewidth=0, alpha=1)
        # plt.scatter(train_diag_vals, gen_diag_vals, color='red', s=8, linewidth=0, label="Same AA", alpha=0.5)
        ax.set_xlabel("True AA Substitution Rate")
        ax.set_ylabel("Gen AA Substitution Rate")
        # ax.legend(loc='upper left', frameon=False, handlelength=0, handletextpad=0)
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=14,
                verticalalignment='top')
    subplots = 6 * 5
    for j in range(subplots - len(alphabet)):
        fig.delaxes(axes.ravel()[subplots - (j + 1)])
    plt.tight_layout()
    fig.savefig(os.path.join(out_path, 'substitution_per_AA.svg'))
    fig.savefig(os.path.join(out_path, 'substitution_per_AA.png'))


    # Plot for all data
    fig, ax = plt.subplots(figsize=(3, 2.5))
    r_squared = stats.pearsonr(train_vals.flatten(), gen_vals.flatten()).statistic
    label = "$R$=%.2f" % (r_squared)
    plt.scatter(train_vals, gen_vals, color='blue', linewidth=0, label="$R^2$=%.2f" % (r_squared), alpha=0.5)
    plt.plot([0, vmax], [0, vmax], linewidth=1, color='black', linestyle="--")
    plt.xlabel("True AA Substitution Rate")
    plt.ylabel("Gen AA Substitution Rate")
    ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    plt.tight_layout()
    fig.savefig(os.path.join(out_path, 'substitution_nondiag.svg'))
    fig.savefig(os.path.join(out_path, 'substitution_nondiag.png'))

    # Plot only same AA substitutions
    fig, ax = plt.subplots(figsize=(3, 2.5))
    r_squared = stats.pearsonr(train_diag_vals, gen_diag_vals).statistic
    label =  "$R$=%.2f" % (r_squared)
    plt.scatter(train_diag_vals, gen_diag_vals, color='red', linewidth=0, label="$R^2$=%.2f" % (r_squared), alpha=1)
    plt.plot([0, vmax], [0, vmax], linewidth=1, color='black', linestyle="--")
    plt.xlabel("True AA Substitution Rate")
    plt.ylabel("Gen AA Substitution Rate")
    ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    plt.tight_layout()
    fig.savefig(os.path.join(out_path, 'substitution_diag.svg'))
    fig.savefig(os.path.join(out_path, 'substitution_diag.png'))

def msa_pairwise_interactions(generated_msa, train_msa, all_aa, out_path):  # Look at AA pairwise interactions within each MSA within each sample
    "Pairwise plots for MSAs"
    all_aa_pairs = list(itertools.product(all_aa, all_aa))
    all_aa_dict = {''.join(k): 1 for k in all_aa_pairs}
    all_aa_dict = {k: all_aa_dict[k] for k in sorted(all_aa_dict.keys())}

    all_pairs_train = get_pairwise(train_msa, all_aa)

    count_map_train = {}
    for i in all_pairs_train:
        i = ''.join(i)
        count_map_train[i] = count_map_train.get(i, 0) + 1

    for aa_pair in all_aa_dict.keys():
        if aa_pair not in count_map_train.keys():
            count_map_train[aa_pair] = 0

    train_dict = {k: count_map_train[k] for k in sorted(count_map_train.keys())}
    total_train = sum(train_dict.values())
    for k in train_dict.keys():
        train_dict[k] = train_dict[k] / total_train

    all_pairs_gen = get_pairwise(generated_msa, all_aa)

    count_map_gen = {}
    for i in all_pairs_gen:
        i = ''.join(i)
        count_map_gen[i] = count_map_gen.get(i, 0) + 1

    for aa_pair in all_aa_dict.keys():
        if aa_pair not in count_map_gen.keys():
            count_map_gen[aa_pair] = 0

    gen_dict = {k: count_map_gen[k] for k in sorted(count_map_gen.keys())}
    total_gen = sum(gen_dict.values())
    for k in gen_dict.keys():
        gen_dict[k] = gen_dict[k] / total_gen

    train_vals = list(train_dict.values())
    gen_vals = list(gen_dict.values())

    r_squared = stats.pearsonr(train_vals, gen_vals).statistic

    fig, ax = plt.subplots(figsize=(3, 2.5))
    label = "$R$=%.2f" % (r_squared)
    plt.plot([0, 0.02], [0, 0.02], linewidth=1, color='black', linestyle="--")
    plt.scatter(train_vals, gen_vals, color='blue', linewidth=0, alpha=0.5)  # marker = alpha
    plt.xlabel("True Parwise Interactions")
    plt.ylabel("Gen Parwise Interactions")
    ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    plt.tight_layout()
    fig.savefig(os.path.join(out_path, 'pairwise.svg'))
    fig.savefig(os.path.join(out_path, 'pairwise.png'))

def plot_tmscores(tmscore_path, out_path, y_min=0, y_max=30):
    "TMscores for conditionally generated sequences, given MSAs"
    tmscores = pd.read_csv(tmscore_path, names=['scores'])
    fig, ax = plt.subplots(figsize=(3, 2.5))
    sns.histplot(tmscores['scores'], color='blue')
    plt.xlabel('TM Scores')
    plt.xlim(0, 1)
    plt.ylim(y_min,y_max)
    plt.tight_layout()
    fig.savefig(os.path.join(out_path, 'tmscores.svg'))
    fig.savefig(os.path.join(out_path, 'tmscores.png'))

def plot_perp_group_masked(df, save_name, mask='mask'):
    "Plot perplexity computed from Masked models, binned by % of sequence masked "
    bins = np.arange(0, 1.1, 0.1)
    df['binned'] = pd.cut(df['time'], bins)
    group = df.groupby(pd.cut(df['time'], bins))
    plot_centers = (bins[:-1] + bins[1:]) / 2
    plot_values = np.exp(group['loss'].sum()/group['tokens'].sum())
    fig, ax = plt.subplots(figsize=(3, 2.5))
    plt.plot(plot_centers*100, plot_values, c='b', marker='o')
    ax.set_xticks([100, 80, 60, 40, 20, 0])
    if mask=='causal-mask':
        plt.gca().invert_xaxis()
        plt.xlabel('% Sequence')
    else:
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        plt.xlabel('% Masked')
    plt.ylabel('Perplexity')
    plt.ylim(0,25)
    plt.tight_layout()
    fig.savefig(os.path.join('plots/perp_'+save_name+'.png'))

def plot_perp_group_d3pm(df, save_name):
    "Plot perplexity computed from D3PM models, binned by timestep intervals"
    bins = np.arange(0, 550, 50)
    df['binned'] = pd.cut(df['time'], bins)
    group = df.groupby(pd.cut(df['time'], bins))
    plot_centers = (bins[:-1] + bins[1:]) / 2
    plot_values = np.exp(group['loss'].sum()/group['tokens'].sum())
    fig, ax = plt.subplots(figsize=(3, 2.5))
    plt.plot(plot_centers, plot_values, c='b', marker='o')
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    plt.xlabel('Timestep')
    plt.ylabel('Perplexity')
    plt.ylim(0, 25)
    plt.tight_layout()
    fig.savefig(os.path.join('plots/perp_' + save_name + '.png'))


def plot_ecdf_bylength(perp_groups, colors, labels, seq_lengths, metric='perp', model='esm-if'):
    "Plots cumulative density as a function of sequence length"
    fig, ax = plt.subplots(1,4, figsize=(8.,2.5), sharey=True, sharex=True)
    for j, perp_group in enumerate(perp_groups):
        for i,p in enumerate(perp_group):
            c=colors[j]
            sns.ecdfplot(x=p,
                         label=labels[j],
                         color=c,
                         alpha=1,
                         ax=ax[i])
            if metric=='perp':
                ax[i].set_xlabel(model+' Perplexity')
            elif metric=='plddt':
                ax[i].set_xlabel(model+' pLDDT')
            ax[i].set_title("seq length="+str(seq_lengths[i]))
            ax[i].axvline(x=np.mean(perp_groups[0][i]), c='k', ls='--', lw=0.75)
    ax[-1].legend(fontsize=8, loc='upper left')
    if model == 'ESM-IF':
        plt.xlim(0, 25)
    elif model == 'MPNN':
        plt.xlim(0, 6)
    elif model == 'Omegafold':
        plt.xlim(10, 100)
    plt.tight_layout()
    fig.savefig(os.path.join('plots/sc_'+metric+'_bylength_'+model+'.svg'))
    fig.savefig(os.path.join('plots/sc_'+metric+'_bylength_'+model+'.png'))

def plot_sc_boxplot(perp_groups, colors, labels, metric='perp', model='ESM-IF', length_model='small', legend=False):
    fig, ax = plt.subplots(1, 1, figsize=(3,3.5), sharey=True, sharex=True)
    all_perp = []
    all_names = []
    all_colors = []
    for i, perp_group in enumerate(perp_groups):
        [all_perp.append(item) for item in list(chain.from_iterable(perp_group))]
        [all_names.append(labels[i]) for _ in range(len(list(chain.from_iterable(perp_group))))]
        all_colors.append(colors[i])

    df = pd.DataFrame()
    df['value'] = all_perp
    df['names'] = all_names
    sns.boxplot(data=df, x="names", y="value", ax=ax, palette=all_colors)

    ax.axhline(y=np.median(list(chain.from_iterable(perp_groups[0]))), c='k', ls='--', lw=0.75)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    if legend:
        ax.legend()
    if model == 'ESM-IF':
        ax.set_ylim(0, 25)
    elif model == 'MPNN':
        ax.set_ylim(0, 6)
    elif model == 'Omegafold':
        ax.set_ylim(10, 100)
    plt.tight_layout()
    fig.savefig(os.path.join('plots/sc_' + metric + '_' + model + '_' + length_model + '.svg'))
    fig.savefig(os.path.join('plots/sc_' + metric + '_' + model + '_' + length_model + '.png'))

def plot_ecdf(perp_groups, colors, labels, metric='perp', model='ESM-IF', length_model='small', legend=False):
    "Plot cumulative density plot of plddt, or perp scores for each set of gen sequences"
    fig, ax = plt.subplots(1,1, figsize=(2.5,2.5), sharey=True, sharex=True)
    for i, perp_group in enumerate(perp_groups):
        c = colors[i]
        all_perp = list(chain.from_iterable(perp_group))
        sns.ecdfplot(x=all_perp,
                         label=labels[i],
                         color=c,
                         alpha=1,
                         ax=ax)
        if metric == 'perp':
            ax.set_xlabel(model + ' Perplexity')
        elif metric == 'plddt':
            ax.set_xlabel(model + ' pLDDT')
        ax.set_title("all sequences")
        ax.axvline(x=np.mean(list(chain.from_iterable(perp_groups[0]))), c='k', ls='--', lw=0.75)
    if legend:
        ax.legend()
    if model=='ESM-IF':
        ax.set_xlim(0,25)
    elif model == 'MPNN':
        ax.set_xlim(0,6)
    elif model == 'Omegafold':
        ax.set_xlim(10, 100)
    plt.tight_layout()
    fig.savefig(os.path.join('plots/sc_'+metric+'_'+model+'_'+length_model+'.svg'))
    fig.savefig(os.path.join('plots/sc_'+metric+'_'+model+'_'+length_model+'.png'))

def plot_plddt_perp(ordered_plddt_group, ordered_perp_group, idx, colors, labels, perp_model='ESM-IF', length_model='small'):
    "Plot pLDDT vs Perplexity for each set of generated sequences against train data"
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5), sharey=True, sharex=True)
    plt.scatter(ordered_plddt_group[0], ordered_perp_group[0], c=colors[0], s=20, alpha=1, label=labels[0], edgecolors='grey')
    plt.scatter(ordered_plddt_group[idx], ordered_perp_group[idx], c=colors[idx], s=20, alpha=1, label=labels[idx], edgecolors='k')
    plt.ylim(0, 25)
    plt.xticks([25, 50, 75, 100])
    ax.set_ylabel(perp_model + ' Perplexity')
    ax.set_xlabel('pLDDT')
    plt.tight_layout()
    fig.savefig(os.path.join('plots/sc_plddt_perp_'+labels[idx]+'_'+length_model+'.svg'))
    fig.savefig(os.path.join('plots/sc_plddt_perp_'+labels[idx]+'_'+length_model+'.png'))

def ss_helix_strand(runs, data, labels, save_name):
    "2D Probability Density plots for DSSP 3-state predictions of % Helix and % Sheet"
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(10, 7), constrained_layout=True, sharex=False, sharey=False)
    ax = ax.ravel()
    for i, run in enumerate(runs):
        helix = data[data['type'] == run]['helix_percent']
        strand = data[data['type'] == run]['strand_percent']

        plt.rcParams['axes.titley'] = 1.0  # y is in axes-relative coordinates.
        plt.rcParams['axes.titlepad'] = -14
        ax[i].set_title(labels[i])

        sns.kdeplot(x=helix, y=strand,
                    fill=True, thresh=0.001, levels=10,
                    cmap='Greys', ax=ax[i], cbar=False, common_norm=True)
        ax[i].set_xlabel('% Helix per Seq')
        ax[i].set_ylabel('% Strand per Seq')
        ax[i].set_xlim(-0.05, 1)
        ax[i].set_ylim(-0.05, 1)
    #plt.tight_layout()
    fig.savefig(os.path.join('plots/helix_strand_' + save_name + '.svg'))
    fig.savefig(os.path.join('plots/helix_strand_' + save_name + '.png'))

def ss_box_whisker(data, colors, save_name):
    "Create box and whisker plot for DSSP 3-state secondary structure predictions"
    fig, ax = plt.subplots(1, 3, figsize=(7, 3.5), sharex=True, sharey=True)
    sns.boxplot(data=data, x="helix_percent", y="type", ax=ax[0], palette=colors)
    sns.boxplot(data=data, x="strand_percent", y="type", ax=ax[1], palette=colors)
    sns.boxplot(data=data, x="other_percent", y="type", ax=ax[2], palette=colors)
    ax[0].set_xlabel('% Helix per Sequence')
    ax[1].set_xlabel('% Strand per Sequence')
    ax[2].set_xlabel('% Loop per Sequence')
    [ax[i].set_ylabel(None) for i in range(len(ax))]
    plt.tight_layout()
    fig.savefig(os.path.join('plots/' + save_name + '_structure_box.svg'))
    fig.savefig(os.path.join('plots/' + save_name + '_structure_box.png'))

def plot_embedding(train_emb, run_emb, colors, i, runs, project_run):
    "Plot embedding space of sequences as 2D TSNE "
    fig, ax = plt.subplots(figsize=(5, 5))
    # Plot test
    plt.scatter(train_emb[:, 0][::10], train_emb[:, 1][::10], s=20, alpha=1, c=colors[0],
                edgecolors='grey')
    # Plot run
    plt.scatter(run_emb[:, 0], run_emb[:, 1], s=20, alpha=0.95,
                c=colors[i+1], edgecolors='k')
    ax.axis('off')
    fig.savefig(os.path.join('plots/fid_' + runs[i+1] + '_' + project_run + '.svg'))
    fig.savefig(os.path.join('plots/fid_' + runs[i+1] + '_' + project_run + '.png'))

def clean_list(list):
    cleanedList = [x for x in list if x ==x]
    return cleanedList

def plot_percent_similarity(all_df, colors, legend=False):
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5), sharey=True, sharex=True)
    #sns.set_palette(sns.color_palette("viridis", len(runs)))
    sns.ecdfplot(all_df, ax=ax, legend=legend, palette=colors)
    #f = sns.boxplot([all_df['Valid MSA'].dropna(), all_df['Cond Max'].dropna(), all_df['Cond Rand'].dropna()],
    #            ax=ax, palette=colors)
    #f.set(xticklabels=['Valid MSA', 'Cond Max', 'Cond Rand'])
    ax.set_xlabel('% Similarity to Original MSA')
    ax.axvline(x=25, c='k', ls='--', lw=0.75)
    ax.set_title("% Sim")
    plt.tight_layout()
    fig.savefig(os.path.join('plots/simmsa.svg'))
    fig.savefig(os.path.join('plots/simmsa.png'))

def plot_conditional_tmscores(tm_df, palette, legend=False, save_path='plots/'):
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5), sharey=True, sharex=True)
    sns.ecdfplot(tm_df, palette=palette, ax=ax, legend=legend)
    ax.set_title("  ")
    ax.axvline(x=0.5, c='k', ls='--', lw=0.75)
    plt.xlim(0,1)
    ax.set_ylabel('CDF')
    ax.set_xlabel('TM Score')
    plt.tight_layout()
    fig.savefig(os.path.join(save_path+'_tmscore.svg'))
    fig.savefig(os.path.join(save_path+'_tmscore.png'))

def plot_conditional_rmsd(pdb, motif_df, out_path='plots/'):
    fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.5))
    ax[0].scatter(motif_df['scaffold_lengths'], motif_df['rmsd'], edgecolors='grey', c='#D0D0D0')
    ax[0].set_xlabel('Scaffold Lengths')
    ax[0].set_ylabel(r'Motif RMSD ($\AA$)')
    ax[1].scatter(motif_df['scores'], motif_df['rmsd'], edgecolors='grey', c='#D0D0D0')
    ax[1].set_xlabel('pLDDT entire sequence')
    ax[1].set_ylabel(r'Motif RMSD ($\AA$)')
    ax[2].scatter(motif_df['scores_fixed'], motif_df['rmsd'], edgecolors='grey', c='#527d99')
    ax[2].set_xlabel('pLDDT fixed region')
    ax[2].set_ylabel(r'Motif RMSD ($\AA$)')
    ax[0].axhline(y=1, c='k', ls='--', lw=0.75)
    ax[1].axhline(y=1, c='k', ls='--', lw=0.75)
    ax[2].axhline(y=1, c='k', ls='--', lw=0.75)
    plt.title("  ")
    ax[1].set_xlim(0, 100)
    ax[2].set_xlim(0, 100)
    plt.tight_layout()
    fig.savefig(os.path.join(out_path + pdb + '.png'))

def plot_conditional_sim(sim, out_path='plots/'):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    sns.histplot(sim, color='grey', bins=10, ax=ax)
    plt.xlabel('% Seq similarity (Fixed)')
    plt.title("  ")
    plt.xlim(0, 100)
    plt.tight_layout()
    fig.savefig(out_path + '_similarity.png')

def idr_parity_plot(mean_og_score, mean_gen_score, out_path):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    r_squared = stats.pearsonr(mean_og_score, mean_gen_score).statistic
    label = "$R$=%.2f" % (r_squared)
    plt.axline([0, 0], [1, 1], c='k', linestyle='dotted', alpha=0.75)
    ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    plt.scatter(mean_og_score, mean_gen_score, c='grey', edgecolors='k')
    plt.xlabel("Per-Res Score True", fontweight='bold')
    plt.ylabel("Per-Res Score Gen", fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_path, 'idr_parity_scatter.svg'))
    fig.savefig(os.path.join(out_path, 'idr_parity_scatter.png'))
    plt.close()

def plot_idr(out_fpath, df, start, end, save_iter):
    fig, ax = plt.subplots(figsize=(6,3))
    plt.plot(df['resid'], df['score'], c='b')
    plt.axhline(y=0.5, c='k', ls='--')
    #plt.axvline(x=end, c='k', ls='--')
    plt.axvspan(start, end, alpha=0.1, color='b')
    plt.ylabel('score')
    plt.xlabel('residue')
    plt.tight_layout()
    fig.savefig(out_fpath+'idr_'+str(save_iter)+'.svg')
    fig.savefig(out_fpath+'idr_'+str(save_iter)+'.png')

def plot_idr_drbert(out_fpath, prefix, df, start, end, save_iter):
    fig, ax = plt.subplots(figsize=(6,3))
    x = np.arange(0,len(df['score'][save_iter]))
    plt.plot(x, df['score'][save_iter], c='b')
    #plt.axhline(y=0.5, c='k', ls='--')
    #plt.axvline(x=end, c='k', ls='--')
    plt.axvspan(start, end, alpha=0.1, color='b')
    plt.ylabel('score')
    plt.xlabel('residue')
    plt.ylim(0,1)
    plt.tight_layout()
    fig.savefig(out_fpath+'svg/'+prefix+str(save_iter)+'.svg')
    #fig.savefig(out_fpath+prefix+str(save_iter)+'.png')


def plot_idr_drbert_multiple(out_fpath, prefix, df, start, end, df2, start2, end2, save_iter):
    fig, ax = plt.subplots(figsize=(4,1.5))
    x = np.arange(0,len(df['score'][save_iter]))
    x2 = np.arange(0,len(df2['score'][save_iter]))
    plt.plot(x, df['score'][save_iter], c='#1E9AC7')
    plt.plot(x2, df2['score'][save_iter], c='grey')
    plt.axvspan(start, end, alpha=0.25, color='#1E9AC7')
    plt.ylabel('score')
    plt.xlabel('residue')
    plt.ylim(0,1)
    plt.tight_layout()
    fig.savefig(out_fpath+'svg/'+prefix+str(save_iter)+'.svg')

def idr_boxplot(gen_disorder_percent, gen_order_percent, out_fpath, save_name):
    fig, ax = plt.subplots(figsize=(3,3))
    f = sns.boxplot([gen_disorder_percent, gen_order_percent], ax=ax)
    f.set(xticklabels=['Disorder', 'Non-Disordered'])
    plt.ylim(0,1)
    plt.tight_layout()
    fig.savefig(out_fpath+save_name+'idr_box.svg')
    fig.savefig(out_fpath+save_name+'idr_box.png')

def idr_boxplot_all(df, out_fpath, save_name):
    print(df)
    fig, ax = plt.subplots(figsize=(3,3))
    f = sns.boxplot(data=df, x="region", y="score", hue='type', ax=ax)
    f.set(xticklabels=['Disorder', 'Non-Disordered'])
    plt.ylim(0,1)
    plt.tight_layout()
    fig.savefig(out_fpath+save_name+'idr_box.svg')
    fig.savefig(out_fpath+save_name+'idr_box.png')
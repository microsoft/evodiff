import pandas as pd
import matplotlib.pyplot as plt
import csv
from collections import Counter, OrderedDict
import torch
from torch.nn import KLDivLoss
import os
import itertools
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import numpy as np
import seaborn as sns
import difflib
from itertools import chain

def removekey(d, list_of_keys):
    r = d.copy()
    for key in list_of_keys:
        del r[key]
    return r

def csv_to_dict(generate_file):
    seqs = ''
    with open(generate_file, 'r') as file:
        filecontent = csv.reader(file)
        for row in filecontent:
            if len(row) >= 1:
                if row[0][0] != '>':
                    seqs += str(row[0])
    aminos_gen = Counter(
        {'A': 0, 'M': 0, 'R': 0, 'T': 0, 'D': 0, 'Y': 0, 'P': 0, 'F': 0, 'L': 0, 'E': 0, 'W': 0, 'I': 0, 'N': 0, 'S': 0, \
         'K': 0, 'Q': 0, 'H': 0, 'V': 0, 'G': 0, 'C': 0, 'X': 0, 'B': 0, 'Z': 0, 'U': 0, 'O': 0, 'J': 0, '-': 0})
    aminos_gen.update(seqs)

    order_of_keys = ['A','M','R','T','D','Y','P','F','L','E','W','I','N','S',
                     'K','Q','H','V','G','C','X','B','Z','J','O','U','-']
    list_of_tuples = [(key, aminos_gen[key]) for key in order_of_keys]
    aminos_gen_ordered = OrderedDict(list_of_tuples)
    return aminos_gen_ordered

def normalize(list):
    norm = sum(list)
    new_list = [item / norm for item in list]
    return new_list

def aa_reconstruction_parity_plot(project_dir, out_path, generate_file, msa=False, idr=False, gen_file=True,
                                  start_valid=False, start_query=False, start_msa=False):
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
    a = normalize(values)  # normalize(list(aminos.values()))
    if start_valid:
        a_kl = normalize(list(removekey(aminos, keys_to_remove).values()))
    else:
        #print(aminos)
        a_kl = normalize([each[0] for each in removekey(aminos, keys_to_remove).values()])
    if gen_file:
        b_list = list(aminos_gen.values())
        b = normalize(b_list) # ADD GAPS IN
        # Save KL to file
        kl_loss = KLDivLoss(reduction="sum")
        if msa:
            b_kl = normalize(list(removekey(aminos_gen, keys_to_remove).values()))
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
        plt.axline([0, 0], [0.1, 0.1], c='k', linestyle='dotted', alpha=0.5)
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

    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    #print(aminos.keys())
    plt.bar(list(aminos.keys())[:-5], a_kl, color='black', alpha=0.5)
    if gen_file:
        plt.bar(list(aminos_gen.keys())[:-6], b_kl, color='b', alpha=0.5)
        ax.text(0.05, 0.95, kl_label, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    plt.xlabel("Amino Acids", fontweight='bold')
    plt.ylabel("Normalized Freq", fontweight='bold')
    plt.tight_layout()
    save_dir_test = os.path.join(out_path)
    fig.savefig(save_dir_test+'/'+idr_flag+gen_flag+'parity_bar.svg')
    fig.savefig(save_dir_test+'/'+idr_flag+gen_flag+'parity_bar.png')
    plt.close()

    if not gen_file:
        return a # return train probability distribution

def get_matrix(all_pairs, all_aa_pairs, alphabet):
    count_map = {}
    for i in all_pairs:
        count_map[i] = count_map.get(i, 0) + (1 / 63)
    for aa_pair in all_aa_pairs:
        if aa_pair not in count_map.keys():
            pass
            count_map[aa_pair] = 0
    _dict = {k: count_map[k] for k in sorted(count_map.keys())}
    _matrix = list(_dict.values())
    _matrix = np.asarray(_matrix).reshape(len(alphabet), len(alphabet))
    return _matrix


def get_pairs(array, alphabet):
    all_pairs = []
    all_q_val = []
    for b in np.arange(array.shape[0]):
        curr_msa = array[b]
        for col in np.arange(curr_msa.shape[1]):
            q_val = curr_msa[0, col]
            if q_val < len(alphabet):
                q_val = curr_msa[0, col]
                all_q_val.append(q_val)
                col_vals = list(curr_msa[1:, col])
                col_vals = filter(lambda val: val < len(alphabet), col_vals)
                curr_pairs = [(q_val, v) for v in col_vals]
                all_pairs.append(curr_pairs)
    all_pairs = list(itertools.chain(*all_pairs))
    return all_pairs


def normalize_matrix(data, alphabet):
    alpha_labels = list(alphabet)
    table = pd.DataFrame(data, index=alpha_labels, columns=alpha_labels)
    table = table / table.sum(axis=0)  # normalize
    table.fillna(0, inplace=True)

    table_vals = table.values
    table_diag_vals = np.diag(table)
    return table, table_vals, table_diag_vals


def msa_substitution_rate(generated_msa, train_msa, alphabet, out_path):
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

def get_pairwise(msa, alphabet):
    all_pairs = []
    queries = msa[:, 0, :]
    for row in queries:
        row = row.astype(int)
        curr_query = list(row[row < len(alphabet)])
        curr_query = [alphabet[c] for c in curr_query if c < len(alphabet)]
        curr_pairs = itertools.permutations(curr_query, 2)
        all_pairs.append(list(curr_pairs))
    all_pairs = list(itertools.chain(*all_pairs))
    return all_pairs

def msa_pairwise_interactions(generated_msa, train_msa, all_aa, out_path):  # Look at AA pairwise interactions within each MSA within each sample

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

    # r2_val = r2_score(train_vals, gen_vals)
    # mse = mean_squared_error(train_vals, gen_vals)
    # with open(out_path + 'generate_metrics.csv', 'a') as f:
    #     f.write("\nR-squared task 3: "+ str(r2_val))
    #     f.write("\nMean squared error task 3: "+str(mse))
    # f.close()
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
    tmscores = pd.read_csv(tmscore_path, names=['scores'])
    fig, ax = plt.subplots(figsize=(3, 2.5))
    sns.histplot(tmscores['scores'], color='blue')
    plt.xlabel('TM Scores')
    plt.xlim(0, 1)
    plt.ylim(y_min,y_max)
    plt.tight_layout()
    fig.savefig(os.path.join(out_path, 'tmscores.svg'))
    fig.savefig(os.path.join(out_path, 'tmscores.png'))

def plot_percent_similarity(out_fpath):
    df_gen = pd.read_csv(out_fpath + 'gen_msas_onlyquery.txt', delim_whitespace=True, header=None,
                         names=['seq'])
    df_valid = pd.read_csv(out_fpath + 'valid_msas_onlyquery.txt', delim_whitespace=True, header=None,
                           names=['seq'])
    # % similarity between original and new query
    sim = []
    for i in range(len(df_gen)):
        s1 = list(itertools.chain.from_iterable(df_gen.iloc[i]['seq']))
        s2 = list(itertools.chain.from_iterable(df_valid.iloc[i]['seq']))
        sm = difflib.SequenceMatcher(None, s1, s2)
        sim.append(sm.ratio())
        # print(df_valid.iloc[i]['seq'].split())

        fig, ax = plt.subplots(figsize=(3, 2.5))
        sns.histplot(sim, color='grey', bins=10)
        plt.xlabel('% Similarity')
        plt.xlim(0, 1)
        plt.tight_layout()
        fig.savefig(os.path.join(out_fpath, 'query_similarity.png'))
        fig.savefig(os.path.join(out_fpath, 'query_similarity.svg'))

def extract_seq_a3m(generate_file):
    list_of_seqs = []
    with open(generate_file, 'r') as file:
            filecontent = csv.reader(file)
            for row in filecontent:
                if len(row) >= 1:
                    if row[0][0] != '>':
                        list_of_seqs.append(str(row[0]))
    return list_of_seqs[1:]

def plot_percent_similarity_entiremsa(out_fpath):
    df_gen = pd.read_csv(out_fpath + 'gen_msas_onlyquery.txt', delim_whitespace=True, header=None,
                         names=['seq'])
    df_valid = pd.read_csv(out_fpath + 'valid_msas_onlyquery.txt', delim_whitespace=True, header=None,
                           names=['seq'])

    # print(df_gen)
    sim = []
    sim_msa = []
    for i in range(len(df_gen)):
        s1 = list(itertools.chain.from_iterable(df_gen.iloc[i]['seq']))
        s2 = list(itertools.chain.from_iterable(df_valid.iloc[i]['seq']))
        sm = difflib.SequenceMatcher(None, s1, s2)
        sim.append(sm.ratio() * 100)
        msa_seqs = extract_seq_a3m(out_fpath + 'gen-' + str(i + 1) + '/generated_msas.a3m')
        for seq in msa_seqs:
            sm = difflib.SequenceMatcher(None, s1, list(itertools.chain.from_iterable(seq)))
            sim_msa.append(sm.ratio() * 100)

    fig, ax = plt.subplots(2, 1, figsize=(3, 2.5))
    sns.histplot(sim_msa, color='grey', bins=20, ax=ax[0])
    ax[0].set_xlabel('% Similarity to seq in MSA')
    sns.histplot(sim, color='grey', bins=20, ax=ax[1])
    ax[1].set_xlabel('% Similarity to original query')
    ax[0].set_xlim(0, 100)
    ax[1].set_xlim(0, 100)
    plt.tight_layout()
    fig.savefig(os.path.join(out_fpath, 'similairity_msa.svg'))
    fig.savefig(os.path.join(out_fpath, 'similarity_msa.png'))

def plot_perp_group_masked(df, save_name):
    bins = np.arange(0, 1.1, 0.1)
    df['binned'] = pd.cut(df['time'], bins)
    group = df.groupby(pd.cut(df['time'], bins))
    plot_centers = (bins[:-1] + bins[1:]) / 2
    plot_values = group['perplexity'].mean()
    plot_std = group['perplexity'].std()

    fig, ax = plt.subplots(figsize=(3, 2.5))
    plt.plot(plot_centers*100, plot_values, c='b')
    plt.errorbar(plot_centers*100, plot_values, yerr=plot_std/2, fmt="o", c='b', capsize=3)
    ax.set_xticks([0,20,40,60,80,100])
    plt.xlabel('% Masked')
    plt.ylabel('Perplexity')
    plt.ylim(0,25)
    plt.tight_layout()
    fig.savefig(os.path.join('plots/perp_'+save_name+'.png'))

def plot_perp_group_d3pm(df, save_name):
    bins = np.arange(0, 550, 50)
    df['binned'] = pd.cut(df['time'], bins)
    group = df.groupby(pd.cut(df['time'], bins))
    plot_centers = (bins[:-1] + bins[1:]) / 2
    plot_values = group['perplexity'].mean()
    plot_std = group['perplexity'].std()

    fig, ax = plt.subplots(figsize=(3, 2.5))
    plt.plot(plot_centers, plot_values, c='b')
    plt.errorbar(plot_centers, plot_values, yerr=plot_std/2, fmt="o", c='b', capsize=3)
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    plt.xlabel('Timestep')
    plt.ylabel('Perplexity')
    plt.ylim(0, 25)
    plt.tight_layout()
    fig.savefig(os.path.join('plots/perp_' + save_name + '.png'))


def plot_ecdf_bylength(perp_groups, colors, labels, seq_lengths, metric='perp', model='esm-if'):
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


def plot_ecdf(perp_groups, colors, labels, metric='perp', model='ESM-IF'):
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
    #ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.7, 1.05),)
    if model=='ESM-IF':
        ax.set_xlim(0,25)
    elif model == 'MPNN':
        ax.set_xlim(0,6)
    elif model == 'Omegafold':
        ax.set_xlim(10, 100)
    plt.tight_layout()
    fig.savefig(os.path.join('plots/sc_'+metric+'_'+model+'.svg'))
    fig.savefig(os.path.join('plots/sc_'+metric+'_'+model+'.png'))

def plot_plddt_perp(ordered_plddt_group, ordered_perp_group, idx, colors, labels, perp_model='ESM-IF'):
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5), sharey=True, sharex=True)
    plt.scatter(ordered_plddt_group[0], ordered_perp_group[0], c=colors[0], s=20, alpha=1, label=labels[0], edgecolors='grey')
    plt.scatter(ordered_plddt_group[idx], ordered_perp_group[idx], c=colors[idx], s=20, alpha=1, label=labels[idx], edgecolors='grey')
    ax.axhline(y=np.mean(ordered_perp_group[0]), c='k', ls='--', lw=0.75)
    ax.axvline(x=np.mean(ordered_plddt_group[0]), c='k', ls='--', lw=0.75)
    plt.ylim(0, 24)
    plt.xticks([25, 50, 75, 100])
    ax.set_ylabel(perp_model + ' Perplexity')
    ax.set_xlabel('pLDDT')
    plt.tight_layout()
    fig.savefig(os.path.join('plots/sc_plddt_perp_'+labels[idx]+'.svg'))
    fig.savefig(os.path.join('plots/sc_plddt_perp_'+labels[idx]+'.png'))

def ss_helix_strand(runs, data, labels, save_name):
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(10, 7.5), constrained_layout=True, sharex=False, sharey=False)
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
    plt.tight_layout()
    fig.savefig(os.path.join('plots/helix_strand_' + save_name + '.svg'))
    fig.savefig(os.path.join('plots/helix_strand_' + save_name + '.png'))

def ss_box_whisker(data, colors, save_name):
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

def plot_embedding(projected_embeddings, colors, i, runs, project_run):
    fig, ax = plt.subplots(figsize=(5, 5))
    # Plot test
    plt.scatter(projected_embeddings[:1000, 0], projected_embeddings[:1000, 1], s=20, alpha=1, c=colors[:1000],
                edgecolors='grey')
    begin = 1000 * (i)
    end = 1000 * (i + 1)
    # Plot run
    plt.scatter(projected_embeddings[begin:end, 0], projected_embeddings[begin:end, 1], s=20, alpha=0.95,
                c=colors[begin:end], edgecolors='grey')
    ax.axis('off')
    fig.savefig(os.path.join('plots/fid_' + runs[i] + '_' + project_run + '.svg'))
    fig.savefig(os.path.join('plots/fid_' + runs[i] + '_' + project_run + '.png'))
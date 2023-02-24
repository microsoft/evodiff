import pandas as pd
import matplotlib.pyplot as plt
import csv
from collections import Counter
import torch
from torch.nn import KLDivLoss
import os
import itertools
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


def plot_training_curves(data_path):
    data = data_path + 'metrics_train.csv'
    data_v = data_path + 'metrics.csv'

    df = pd.read_csv(data, names=['loss', 'nll', 'acc', 'tokens', 'step', 'epoch'])
    df_v = pd.read_csv(data_v, names=['loss', 'nll', 'acc', 'tokens', 'step', 'epoch'])

    x = df['step']
    x_v = df_v['step']

    fig, ax = plt.subplots(3, 1, figsize=(3.5, 6), sharex=True)
    ax[0].plot(x, df.loss, c='b', alpha=1)
    ax[1].plot(x, df.nll, c='b', alpha=1, label='train')
    ax[2].plot(x, df.acc, c='b', alpha=1, label='train')

    ax[0].plot(x_v, df_v.loss, c='b', alpha=0.5)
    ax[1].plot(x_v, df_v.nll, c='b', alpha=0.5, label='train')
    ax[2].plot(x_v, df_v.acc, c='b', alpha=0.5, label='train')

    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('NLL')
    ax[2].set_ylabel('Accu')
    ax[2].set_xlabel('Time')
    plt.show()
    # TODO save figure or delete function if not using

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
    return aminos_gen

def normalize(list):
    norm = sum(list)
    new_list = [item / norm for item in list]
    return new_list

def aa_reconstruction_parity_plot(project_dir, out_path, generate_file, msa=False):
    # Load in approx train distribution
    if msa:
        file = project_dir + 'ref/openfold_ref.csv'
    else:
        file = project_dir + 'ref/uniref50_aa_ref.csv' # TODO add file to git
    df = pd.read_csv(file)
    aminos = df.to_dict('list')
    values = [each[0] for each in aminos.values()]

    # Load in generated seqs and count values
    generate_file = out_path + generate_file
    aminos_gen = csv_to_dict(generate_file)

    # Normalize scores
    a = normalize(values)  # normalize(list(aminos.values()))
    b = normalize(list(aminos_gen.values()))

    # Plot
    colors = ['black', 'grey', 'lightcoral', 'brown', 'tomato', 'peru',
              'darkorange', 'goldenrod', 'khaki', 'olive', 'yellow', 'olivedrab',
              'yellowgreen', 'palegreen', 'forestgreen', 'turquoise', 'paleturquoise',
              'cyan', 'deepskyblue', 'dodgerblue', 'royalblue', 'navy', 'blue',
              'darkslateblue', 'mediumpurple', 'darkviolet', 'violet', 'mediumvioletred',
              'crimson', 'lightpink']
    fig = plt.figure(figsize=(3, 2.5))
    annotations = list(aminos_gen.keys())[0:26]
    plt.axline([0, 0], [0.1, 0.1], c='k', linestyle='dotted', alpha=0.5)
    for i, label in enumerate(annotations):
        plt.scatter(a[i], b[i], label=label, c=colors[i], edgecolors='k')
    plt.xlabel("Test Freq", fontweight='bold')
    plt.ylabel("Gen Freq", fontweight='bold')
    plt.tight_layout()
    save_dir_test = os.path.join(out_path, 'parity_scatter.png')
    fig.savefig(save_dir_test)
    plt.close()

    fig = plt.figure(figsize=(4.5, 2.5))
    plt.bar(aminos.keys(), a, color='black', alpha=0.5)
    plt.bar(aminos_gen.keys(), b, color='b', alpha=0.5)
    plt.xlabel("Amino Acids", fontweight='bold')
    plt.ylabel("Normalized Freq", fontweight='bold')
    plt.tight_layout()
    save_dir_test = os.path.join(out_path, 'parity_bar.png')
    fig.savefig(save_dir_test)
    plt.close()

    kl_loss = KLDivLoss(reduction="sum")
    if msa:
        kl = kl_loss(torch.tensor(a[:-6]+[a[-1]]).log(),torch.tensor(b[:-6]+[b[-1]]))
    else:
        kl = kl_loss(torch.tensor(a[0:26]).log(), torch.tensor(b[0:26]))
    print("KL Loss", kl)

    with open(out_path + 'generate_metrics.csv', 'w') as f:
        f.write("aa freq kl:" +str(kl))
    f.close()

def get_pairs(array, alphabet):
    all_pairs = []
    all_q_val = []
    for b in np.arange(array.shape[0]):
        curr_msa = array[b]
        for col in np.arange(curr_msa.shape[1]):
            q_val = curr_msa[0, col]
            all_q_val.append(q_val)
            col_vals = list(curr_msa[1:, col])
            col_vals = filter(lambda val: val < len(alphabet), col_vals)
            curr_pairs = [(q_val, v) for v in col_vals]
            all_pairs.append(curr_pairs)
    all_pairs = list(itertools.chain(*all_pairs))
    return all_pairs

def msa_substitution_rate(generated_msa, train_msa, alphabet, out_path):
    print(len(alphabet))
    all_aa = np.arange(len(alphabet))
    all_aa_pairs = list(itertools.product(all_aa, all_aa))

    all_pairs_train = get_pairs(train_msa, alphabet)

    count_map_train = {}
    for i in all_pairs_train:
        count_map_train[i] = count_map_train.get(i, 0) + (1 / 63)
    for aa_pair in all_aa_pairs:
        if aa_pair not in count_map_train.keys():
            count_map_train[aa_pair] = 0
    train_dict = {k: count_map_train[k] for k in sorted(count_map_train.keys())}
    train_matrix = list(train_dict.values())
    train_matrix = np.asarray(train_matrix).reshape(len(alphabet), len(alphabet))

    alpha_labels = list(alphabet)
    train_table = pd.DataFrame(data=train_matrix.T, index=alpha_labels, columns=alpha_labels)
    non_zero_cols_train = train_table.columns[(train_table != 0).any()]
    train_table[non_zero_cols_train] = train_table[non_zero_cols_train] / train_table[non_zero_cols_train].sum()

    train_vals = train_table.values.tolist()
    train_vals = list(itertools.chain(*train_vals))
    train_diag_vals = np.diag(train_table)

    all_pairs_gen = get_pairs(generated_msa, alphabet)

    count_map_gen = {}
    for i in all_pairs_gen:
        count_map_gen[i] = count_map_gen.get(i, 0) + (1 / 63)

    for aa_pair in all_aa_pairs:
        if aa_pair not in count_map_gen.keys():
            count_map_gen[aa_pair] = 0

    gen_dict = {k: count_map_gen[k] for k in sorted(count_map_gen.keys())}
    gen_matrix = list(gen_dict.values())
    gen_matrix = np.asarray(gen_matrix).reshape(len(alphabet), len(alphabet))

    gen_table = pd.DataFrame(data=gen_matrix.T, index=alpha_labels, columns=alpha_labels)
    non_zero_cols_gen = gen_table.columns[(gen_table != 0).any()]
    gen_table[non_zero_cols_gen] = gen_table[non_zero_cols_gen] / gen_table[non_zero_cols_gen].sum()

    gen_vals = gen_table.values.tolist()
    gen_vals = list(itertools.chain(*gen_vals))
    gen_diag_vals = np.diag(gen_table)

    # Save to file
    r2_val = r2_score(train_vals, gen_vals)
    mse = mean_squared_error(train_vals, gen_vals)
    r2_val_diag = r2_score(train_diag_vals, gen_diag_vals)
    mse_diag = mean_squared_error(train_diag_vals, gen_diag_vals)
    with open(out_path + 'generate_metrics.csv', 'a') as f:
        f.write("\nR-squared task 2: "+str(r2_val))
        f.write("\nMean squared error task 2: "+str(mse))
        f.write("\nR-squared diagonal task 2: "+str(r2_val_diag))
        f.write("\nMean squared error diagonal task 2: "+str(mse_diag))
    f.close()

    # Save plot
    fig = plt.figure(figsize=(3, 2.5))
    plt.scatter(train_vals, gen_vals, color='blue', s=8, linewidth=0, label="Different AA", alpha=0.25)
    plt.scatter(train_diag_vals, gen_diag_vals, color='red', s=8, linewidth=0, label="Same AA", alpha=0.5)
    plt.plot([0, 0.5], [0, 0.5], linewidth=1, color='black', linestyle="--")
    plt.xlabel("True AA Substitution Rate")
    plt.ylabel("Gen AA Substitution Rate")
    plt.tight_layout()
    save_dir_test = os.path.join(out_path, 'substitution.png')
    fig.savefig(save_dir_test)

def get_pairwise(msa, alphabet):
    all_pairs = []
    queries = msa[:, 0, :]
    for row in queries:
        row = row.astype(int)
        curr_query = list(row[row <= len(alphabet)])
        curr_query = [alphabet[c] for c in curr_query]
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

    r2_val = r2_score(train_vals, gen_vals)
    mse = mean_squared_error(train_vals, gen_vals)
    with open(out_path + 'generate_metrics.csv', 'a') as f:
        f.write("\nR-squared task 3: "+ str(r2_val))
        f.write("\nMean squared error task 3: "+str(mse))
    f.close()

    fig = plt.figure(figsize=(3, 2.5))
    plt.scatter(train_vals, gen_vals, color='blue', linewidth=0, s=8, alpha=0.5)  # marker = alpha
    plt.plot([0, 0.02], [0, 0.02], linewidth=1, color='black', linestyle="--")
    plt.xlabel("True Parwise Interactions")
    plt.ylabel("Gen Parwise Interactions")
    plt.tight_layout()
    save_dir_test = os.path.join(out_path, 'pairwise.png')
    fig.savefig(save_dir_test)
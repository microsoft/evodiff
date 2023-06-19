import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import kde
import re
from analysis.plot import plot_ecdf_bylength, plot_ecdf, plot_plddt_perp
from itertools import chain

def get_files(path):
    files = []
    for filename in os.listdir(path):
        if filename.startswith("sequence_scores"):
            files.append(os.path.join(path, filename))
    return files

def get_pdb(path):
    files = []
    for filename in os.listdir(path):
        if filename.endswith(".pdb"):
            files.append(os.path.join(path, filename))
    return files

def get_mpnn(path):
    files = []
    for filename in os.listdir(path):
        if len(os.listdir(os.path.join(path, filename+'/scores/'))) > 0:
            #print(path, filename)
            sub_file = os.listdir(os.path.join(path, filename+'/scores/'))[0]
            #print(sub_file)
            sub_file_path = os.path.join(path+filename+'/scores/', sub_file)
            #print(sub_file_path)
            if os.path.exists(sub_file_path):
                #print(os.path.join(path, filename+'/scores', sub_f)
                files.append(os.path.join(path+filename+'/scores/', sub_file))
    return files

def get_perp(files):
    perplexity = []
    perp_index = []
    for f in files:
        perp_index.append(re.findall('\d+', f)[-1])
        temp = pd.read_csv(f, header=None, usecols=[1], names=['perp'])
        if temp.empty:
            print("perp", f)
        else:
            #print(f)
            #print(temp)
            perplexity.append(temp.perp[0])
    return perplexity, perp_index

def get_confidence_score(files):
    #colors = {32:'red', 64:'orange', 128:'green', 256:'b', 384:'purple', 512:'grey', 1024:'k'}
    scores = []
    #lengths = []
    pdb_index = []
    for f in files:
        #print(f)
        # Get pdb file number
        pdb_index.append(re.findall('\d+', f)[-1])
        df = pd.read_csv(f, delim_whitespace=True, header=None, usecols=[5,10], names=['residue','score'])
        df = df.dropna() # ignore empty rows
        #print(df)
        if df.empty:
            print("confidence empty", f)
        else:
            if "C" in str(df.score):
                print(df[df.isin(["C"])])
                print("confidence", f)
                print(df.score.mean())
                #print(df.tail())
                #import pdb; pdb.set_trace()
            else:
                #print(df)
                key = int(df.iloc[-1]['residue']+1)
                #print(key)
                #closest_key = min(list(colors.keys()), key = lambda x: abs(x-key))
                #print(key, closest_key)
                #lengths.append(colors[closest_key])
                scores.append(df.score.mean())
    return scores, pdb_index

def get_mpnn_scores(files):
    scores = []
    for f in files:
        #print(f)
        d = np.load(f)
        #print(d['score']) # average over residues that were designed negative log probability of sampled amino acids
        #print(    d['global_score']) # same but average over all chains
        scores.append(np.exp(d['score'][0]))
    return scores

def iterate_dirs(run, seq_lengths, mpnn=False):
    # seq_lengths = [32, 64, 128, 256] #, 512] #, 384] #, 512]
    perp_group = []
    scores_group = []
    lengths_group = []
    mpnn_scores_group = []
    pdb_index_group = []
    perp_index_group = []

    for l in seq_lengths:
        path = '/home/v-salamdari/Desktop/DMs/blobfuse/' + str(run) + 'esmif/' + str(l) + '/'
        if mpnn:
            mpnn_path = '/home/v-salamdari/Desktop/DMs/blobfuse/' + str(run) + 'mpnn/' + str(l) + '/'
        pdb_path = '/home/v-salamdari/Desktop/DMs/blobfuse/' + str(run) + 'pdb/' + str(l) + '/'
        # Get ESMIF perp
        files = get_files(path)
        perplexity, perp_index = get_perp(files)
        perp_group.append(perplexity)
        perp_index_group.append(perp_index)

        # Get pdb
        pdb_files = get_pdb(pdb_path)
        score, pdb_index = get_confidence_score(pdb_files)
        scores_group.append(score)
        #lengths_group.append(lengths)
        pdb_index_group.append(pdb_index)
        # Get MPNN score
        if mpnn:
            mpnn_files = get_mpnn(mpnn_path)
            mpnn_scores = get_mpnn_scores(mpnn_files)
            mpnn_scores_group.append(mpnn_scores)

    all_perp = list(chain.from_iterable(perp_group))
    #print("esmif mean", np.mean(all_perp))
    if mpnn:
        all_mpnn = list(chain.from_iterable(mpnn_scores_group))
        #print("mpnn mean", np.mean(all_mpnn))
    all_scores = list(chain.from_iterable(scores_group))
    #print("omegafold mean", np.mean(all_scores))
    if mpnn:
        return perp_group, scores_group, lengths_group, mpnn_scores_group, pdb_index_group, perp_index_group
    else:
        return perp_group, scores_group, lengths_group, pdb_index_group, perp_index_group

def mean_metric(groups, metric='perp'):
    "Get mean of metric"
    for i in range(len(groups)):
        all = list(chain.from_iterable(groups[i]))
        print(labels[i], np.mean(all))

# Iterate over mdoels
length_model='large'
# TEST MUST GO FIRST FOR PLOTS TO REFERENCE CORRECTLY
runs = ['test-data/', 'd3pm/blosum-640M-0/', 'd3pm/random-640M-0/', 'd3pm/oaardm-640M/', 'd3pm/soar-640M/',
        'hyper12/cnn-650M/', 'esm-1b/','esm2/', 'random-ref/']

#length_model='small'
# runs = ['test-data/','sequence/blosum-0-seq/', 'd3pm-final/random-0-seq/', 'sequence/oaardm/', 'arcnn/cnn-38M/',
#         'pretrain21/cnn-38M/', 'random-ref/']

labels=['Test', 'D3PM Blosum', 'D3PM Uniform', 'OA-AR', 'LR-AR', 'CARP', 'ESM-1b', 'ESM2', 'Random']
mpnn=False # If you also ran MPNN

perp_groups = []
scores_groups = []
lengths_groups = []
mpnn_scores_groups = []
pdb_index_groups = []
perp_index_groups = []
seq_lengths = [64, 128, 256, 384]

for run in runs:
    print("run", run)
    if run == 'esm2/':
        perp_group, scores_group, lengths_group, pdb_index_group, perp_index_group = iterate_dirs(run, [100],
                                                                                                  mpnn=mpnn)
    else:
        if mpnn:
            perp_group, scores_group, lengths_group, mpnn_scores_group, pdb_index_group, perp_index_group = iterate_dirs(run, seq_lengths, mpnn=mpnn)
        else:
            perp_group, scores_group, lengths_group, pdb_index_group, perp_index_group = iterate_dirs(run, seq_lengths, mpnn=mpnn)
    perp_groups.append(perp_group)
    scores_groups.append(scores_group)
    lengths_groups.append(lengths_group)
    pdb_index_groups.append(pdb_index_group)
    perp_index_groups.append(perp_index_group)
    if mpnn:
        mpnn_scores_groups.append(mpnn_scores_group)

colors = ['#D0D0D0', "#b0e16d", '#63C2B5', '#46A7CB', '#1B479D', 'plum', 'mediumpurple', 'rebeccapurple', 'darkslateblue', 'firebrick']

#For ESM-IF
print("ESM-IF")
plot_ecdf_bylength(perp_groups, colors, labels, seq_lengths, metric='perp', model='ESM-IF')
plot_ecdf(perp_groups, colors, labels, model='ESM-IF')
mean_metric(perp_groups, metric='perp')

# For MPNN
if mpnn:
    print("MPNN")
    plot_ecdf_bylength(mpnn_scores_groups, colors, labels, seq_lengths, metric='perp', model='MPNN')
    plot_ecdf(mpnn_scores_groups, colors, labels, model='MPNN')
    mean_metric(mpnn_scores_groups, metric='perp')

print("Omegafold")
# For Omegafold
plot_ecdf_bylength(scores_groups, colors, labels, seq_lengths, metric='plddt', model='Omegafold')
plot_ecdf(scores_groups, colors, labels, metric='plddt', model='Omegafold')
mean_metric(scores_groups, metric='plddt')

# Organize plddt and perp by pdb index
ordered_perp_group = []
ordered_plddt_group = []

for i in range(len(labels)):
    ordered_perp = []
    ordered_plddt = []
    if labels[i] == 'ESM2':
        seq_lengths=[100]
    else:
        seq_lengths= [64, 128, 256, 384]
    for l_index in range(len(seq_lengths)):
        df_pdb = pd.DataFrame(np.array([list(map(float, pdb_index_groups[i][l_index])), \
                                        list(map(float, scores_groups[i][l_index]))]).T, columns=['pdb', 'plddt'])
        df_perp = pd.DataFrame(np.array([list(map(float, perp_index_groups[i][l_index])), \
                                         list(map(float, perp_groups[i][l_index]))]).T, columns=['pdb', 'perp'])
        df = pd.merge(df_pdb, df_perp, on=['pdb'], how='left')
        ordered_plddt += list(df['plddt'])
        ordered_perp += list(df['perp'])

    ordered_perp_group.append(ordered_perp)
    ordered_plddt_group.append(ordered_plddt)
print("Len of ordered array", len(ordered_perp_group))

"PLDDT AND PERP"
mean_train_score = np.mean(list(chain.from_iterable(scores_groups[0])))
mean_train_perp = np.mean(list(chain.from_iterable(perp_groups[0])))
# Plot PLDDT vs perp for all models
for idx in range(len(labels)):
    if idx>0:
        plot_plddt_perp(ordered_plddt_group, ordered_perp_group, idx, colors, labels, perp_model='ESM-IF')
    c_df = pd.DataFrame(np.array([ordered_plddt_group[idx], ordered_perp_group[idx]]).T, columns=['plddt', 'perp'])
    #print(c_df[c_df['perp'] <= mean_train_perp])
    print(labels[idx],
          len(c_df[c_df['perp'] <= mean_train_perp]),
          len(c_df[c_df['plddt'] >= mean_train_score]),
          sum(c_df[c_df['perp'] <= mean_train_perp]['plddt'] >= mean_train_score))
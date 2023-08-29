import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from sequence_models.datasets import TRRMSADataset, A3MMSADataset
from torch.utils.data import DataLoader
from evodiff.utils import Tokenizer
from evodiff.collaters import D3PMCollater
import itertools
import difflib
import seaborn as sns
import pandas as pd
from evodiff.plot import plot_percent_similarity, plot_conditional_tmscores


def extract_seq_a3m(generate_file):
    list_of_seqs = []
    with open(generate_file, 'r') as file:
            filecontent = csv.reader(file)
            for row in filecontent:
                if len(row) >= 1:
                    if row[0][0] != '>':
                        list_of_seqs.append(str(row[0]))
    return list_of_seqs[1:]

def calc_sim(df_gen, df_valid, path_to_file, verbose=True):
    sim = []
    sim_msa = []
    included_list = [106, 102, 75, 28, 84]
    msa_seqs = pd.read_csv(path_to_file + run + 'valid_msas_onlymsa.txt', delim_whitespace=True, header=None, names=['seq'])
    for i in range(len(df_gen)):
        s1 = list(itertools.chain.from_iterable(df_gen.iloc[i]['seq']))
        s2 = list(itertools.chain.from_iterable(df_valid.iloc[i]['seq']))
        sm=difflib.SequenceMatcher(None,s1,s2)
        sim.append(sm.ratio()*100)
        #msa_seqs = extract_seq_a3m(path_to_file+run+'gen-'+str(i+1)+'/generated_msas.a3m')
        start=i*63
        end = (i+1)*63
        #print(msa_seqs.shape)
        #print(i, start, end, len(msa_seqs))
        #import pdb ; pdb.set_trace()
        #print(msa_seqs[start:end])
        #if i >= 1: # for seqs in msa (not including query)
        for index, seq in msa_seqs[start:end].iterrows():
            #print(seq)
            #print(list(itertools.chain.from_iterable(seq)))
            sm_msa=difflib.SequenceMatcher(None,s1,list(itertools.chain.from_iterable(seq)))
            sim_msa.append(sm_msa.ratio()*100)
        if verbose:
            if i in included_list:
                print(i, sm.ratio()*100, sm_msa.ratio()*100)
                #print("gen", "".join(s1))
                #print("valid", "".join(s2))
    return sim, sim_msa

path_to_file = 'amlt-generate-msa/'
runs = ['msa-oaardm-max-train-startmsa/', 'msa-oaardm-max-train-startmsa/',
        #'msa-oaardm-random-train-startmsa/', '
        'msa-esm-startmsa-t2/','potts/']
labels = ['Valid', 'Cond Max', #'Cond Rand',
          'ESM-1b','Potts']
colors = ['#D0D0D0'] + sns.color_palette("viridis", len(runs)-1) #+ ['#D0D0D0']
palette = {labels[i]: colors[i] for i in range(len(labels))}

for i, run in enumerate(runs):
    print(run)
    df_gen = pd.read_csv(path_to_file + run + 'gen_msas_onlyquery.txt', delim_whitespace=True, header=None, names=['seq'])
    df_valid = pd.read_csv(path_to_file + run + 'valid_msas_onlyquery.txt', delim_whitespace=True, header=None, names=['seq'])
    if i == 0: # append valid 1x
        sim, sim_msa = calc_sim(df_valid, df_valid, path_to_file, verbose=False)
        all_df = pd.DataFrame(sim_msa, columns=[labels[i]]) # only want query to msa sim for valid
    else:
        sim, sim_msa = calc_sim(df_gen, df_valid, path_to_file)
        new = pd.DataFrame(np.append(np.array(sim), np.array(sim_msa)), columns=[labels[i]])
        all_df = pd.concat([all_df, new], axis=1)
    #print(all_df.tail())
print([(label, all_df[label].mean()) for label in labels])
plot_percent_similarity(all_df, colors, legend=True)

all_tm = []
for i, run in enumerate(runs[1:]):
    tm = pd.read_csv(path_to_file + run + 'tmscores.txt', names=[labels[1:][i]])
    print(len(tm))
    all_tm.append(tm)
tm_df = pd.concat(all_tm, axis=1)
print(tm_df.head())
plot_conditional_tmscores(tm_df, palette, legend=True)
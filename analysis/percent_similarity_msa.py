import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from sequence_models.datasets import TRRMSADataset, A3MMSADataset
from torch.utils.data import DataLoader
from dms.utils import Tokenizer
from dms.collaters import D3PMCollater
import itertools
import difflib
import seaborn as sns
import pandas as pd
from analysis.plot import plot_percent_similarity, plot_tmscore


def extract_seq_a3m(generate_file):
    list_of_seqs = []
    with open(generate_file, 'r') as file:
            filecontent = csv.reader(file)
            for row in filecontent:
                if len(row) >= 1:
                    if row[0][0] != '>':
                        list_of_seqs.append(str(row[0]))
    return list_of_seqs[1:]

def calc_sim(df_gen, df_valid, path_to_file):
    sim = []
    sim_msa = []
    msa_seqs = pd.read_csv(path_to_file + run + 'valid_msas_onlymsa.txt', delim_whitespace=True, header=None, names=['seq'])
    for i in range(len(df_gen)):
        s1 = list(itertools.chain.from_iterable(df_gen.iloc[i]['seq']))
        s2 = list(itertools.chain.from_iterable(df_valid.iloc[i]['seq']))
        sm=difflib.SequenceMatcher(None,s1,s2)
        sim.append(sm.ratio()*100)
        #msa_seqs = extract_seq_a3m(path_to_file+run+'gen-'+str(i+1)+'/generated_msas.a3m')
        start=i*63
        end = (i+1)*63
        #print(i, start, end, len(msa_seqs))
        #print(msa_seqs[start:end])
        for index, seq in msa_seqs[start:end].iterrows():
            #print(seq)
            #print(list(itertools.chain.from_iterable(seq)))
            sm=difflib.SequenceMatcher(None,s1,list(itertools.chain.from_iterable(seq)))
            sim_msa.append(sm.ratio()*100)
    return sim, sim_msa

path_to_file = '../DMs/amlt-generate-msa/'
runs = ['msa-oaardm-max-train-startmsa/', 'msa-oaardm-random-train-startmsa/', 'msa-esm-startmsa-t2/','potts/']
labels = ['Cond Max', 'Cond Rand', 'ESM-1b','Potts']

all_sim = []
all_msa = []
for run in runs:
    print(run)
    df_gen = pd.read_csv(path_to_file + run + 'gen_msas_onlyquery.txt', delim_whitespace=True, header=None, names=['seq'])
    df_valid = pd.read_csv(path_to_file + run + 'valid_msas_onlyquery.txt', delim_whitespace=True, header=None, names=['seq'])

    sim, sim_msa = calc_sim(df_gen, df_valid, path_to_file)
    all_sim.append(sim)
    all_msa.append(sim_msa)

all_df = pd.DataFrame(np.append(np.array(all_sim), (np.array(all_msa)), axis=1).T, columns=labels)
print([(label, all_df[label].mean()) for label in labels])

#all_sim_df = pd.DataFrame(np.array(all_sim).T, columns=labels)
#all_msa_df = pd.DataFrame(np.array(all_msa).T, columns=labels)
plot_percent_similarity(runs, all_df)

all_tm = []
for i, run in enumerate(runs):
    tm = pd.read_csv(path_to_file + run + 'tmscores.txt', names=['tmscore_'+labels[i]])
    print(len(tm))
    all_tm.append(tm)
#print(all_tm)
#tm_df = pd.DataFrame(np.array(all_tm).T)
tm_df = pd.concat(all_tm, axis=1)
print(tm_df.head())
plot_tmscore(tm_df, legend=False)
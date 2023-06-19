import numpy as np
from bio_embeddings.project import tsne_reduce
import matplotlib.pyplot as plt
from bio_embeddings.embed import ProtTransBertBFDEmbedder, ESM1bEmbedder
import pandas as pd
import csv
import os
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
from dms.utils import parse_txt

# Need to run PGP first on generated seqs , this performs downstream analysis
# https://github.com/hefeda/PGP


def calculate_fid(act1, act2):
    """calculate frechet inception distance"""
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Calculate FID between test dataset sample and generated seqs
project_dir = '../PGP/PGP_OUT_LARGE/'
project_run='large' #'small'
test_fasta = project_dir + 'test/seqs.txt'
test_sequences = parse_txt(test_fasta)

runs = ['blosum', 'random', 'oaardm', 'soardm', 'carp', 'ref', 'valid', 'esm-1b', 'esm2']
c = ["#b0e16d", '#63C2B5', '#46A7CB', '#1B479D', 'lightcoral', 'firebrick', 'grey', 'mediumpurple', 'rebeccapurple']

sequences = test_sequences
colors = ['#D0D0D0'] * len(sequences)
for i, run in enumerate(runs):
    gen_file = project_dir + run + '/seqs.txt'
    gen_df = pd.read_csv(gen_file, names=['sequences'])
    gen_sequences = list(gen_df.sequences)
    [sequences.append(s) for s in gen_sequences]
    [colors.append(c[i]) for s in gen_sequences]

runs = ['test'] + runs
runs

# Can use any embedding here to do the same analysis
embedder = ProtTransBertBFDEmbedder()
embeddings = embedder.embed_many([s for s in sequences])
embeddings = list(embeddings)
reduced_embeddings = [ProtTransBertBFDEmbedder.reduce_per_protein(e) for e in embeddings]
#print(reduced_embeddings[0].shape, reduced_embeddings[1].shape)

options = {
    'n_components':2, # Can use 2 or 3 for best visualization
    'perplexity': 3,
    'n_iter': 500
}
projected_embeddings = tsne_reduce(reduced_embeddings, **options)
#print(projected_embeddings.shape)

# Plot and save to file
for i in range(1, len(runs)):
    fig, ax = plt.subplots(figsize=(5, 5))
    # Plot test
    plt.scatter(projected_embeddings[:1000,0], projected_embeddings[:1000,1], s=20, alpha=1, c=colors[:1000], edgecolors='grey')
    begin=1000*(i)
    end=1000*(i+1)
    # Plot run
    plt.scatter(projected_embeddings[begin:end,0], projected_embeddings[begin:end, 1], s=20, alpha=0.95, c=colors[begin:end], edgecolors='grey')
    ax.axis('off')
    #plt.show()
    #print(runs[i])
    fig.savefig(os.path.join('plots/fid_'+runs[i]+'_'+project_run+'.svg'))
    fig.savefig(os.path.join('plots/fid_'+runs[i]+'_'+project_run+'.png'))

# Calculate FID
reduced_embeddings_by_model = np.array(reduced_embeddings).reshape(len(runs),-1,1024) # 7 runs x 300 sample x 1024 params
fids = []
for i in range(len(reduced_embeddings_by_model)): # compare all to test
    curr_fid = calculate_fid(reduced_embeddings_by_model[0], reduced_embeddings_by_model[i])
    #ref_fid = calculate_fid(reduced_embeddings_by_model[-1], reduced_embeddings_by_model[i])
    fids.append(curr_fid)
    #print "%, " % (runs[i-1], curr_fid)
    print(f'{runs[i]} to test, {curr_fid : 0.2f}')
    #print(f'{runs[i]} to ref , {ref_fid : 0.2f}')
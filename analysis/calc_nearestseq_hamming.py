import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import seaborn as sns
from Bio import Align
from scipy.spatial import distance
from tqdm import tqdm
import torch
from dms.utils import Tokenizer
import math

tokenizer = Tokenizer()

def tokenize_fasta(fasta_file):
    seqs = []
    with open(fasta_file, 'r') as file:
        filecontent = csv.reader(file)
        for row in filecontent:
            if len(row) >= 1:
                if row[0][0] != '>':
                    padded_row = list(row[0])
                    tokenized = [tokenizer.tokenize(s).item() for s in padded_row]
                    seqs.append(tokenized)
    return seqs

def parse_train(fasta_file):
    "Get all sequences of the same length from train dataset"

    seq_lengths = [64, 128, 256, 384]
    seqs_64 = []
    seqs_128 = []
    seqs_256 = []
    seqs_384 = []
    with open(fasta_file, 'r') as file:
        filecontent = csv.reader(file)
        for row in tqdm(filecontent):
            if len(row) >= 1:
                if row[0][0] != '>':
                    if len(row[0]) == seq_lengths[0] or len(row[0]) == seq_lengths[1] or len(row[0]) == seq_lengths[2] or len(row[0]) == seq_lengths[3]:
                        padded_row = list(row[0])
                        tokenized = [tokenizer.tokenize(s).item() for s in padded_row]
                        if len(row[0]) == seq_lengths[0]:
                            seqs_64.append(tokenized)
                        elif len(row[0]) == seq_lengths[1]:
                            seqs_128.append(tokenized)
                        elif len(row[0]) == seq_lengths[2]:
                            seqs_256.append(tokenized)
                        elif len(row[0]) == seq_lengths[3]:
                            seqs_384.append(tokenized)
    return seqs_64, seqs_128, seqs_256, seqs_384

def batch_hamming(train_set, g):
    "Compute hamming distance for a batch of data"
    all_dist = [distance.hamming(t, g) for t in train_set]
    return all_dist

project_dir = '../DMs/'
train_fasta = project_dir + 'data/uniref50/' + 'consensus.fasta'
train_seqs = parse_train(train_fasta) # takes ~ 4 mim
[len(train_seqs[i]) for i in range(4)] # Num seqs evaluated

per_length = 250
num_lengths = 4 # ~ 20 min per length
batch_size = 10000

# Compute Hamming between all natural sequences
train_dists = []
min_dist = 1
for i in range(num_lengths):
    num_batches = math.ceil(len(train_seqs[i])/batch_size)
    for batch in range(num_batches):
        seq_arr=np.array([np.array(s) for s in train_seqs[i][batch*batch_size:(batch+1)*batch_size]])
        print("batch", batch, "of", num_batches, "seq arr", seq_arr.shape)
        all_dist = list(distance.pdist(np.array(seq_arr), metric='hamming'))
        if min(all_dist) <= min_dist:
            min_dist = min(all_dist)
            print("minimum dis", min_dist)

# Calculate each dist to train
project_dir = '../DMs/'
runs = ['sequence/blosum-0-seq/', 'sequence/oaardm/', 'd3pm-final/random-0-seq', 'arcnn/cnn-38M/', 'pretrain21/cnn-38M/', 'esm-1b/']
all_mins = []

for run in runs:
    gen_fasta = project_dir + 'blobfuse/'+run+'generated_samples_string.fasta'
    seqs = tokenize_fasta(gen_fasta)

    per_length = 250
    num_lengths = 4  # ~ 20 min per length

    batch_size = 10000

    train_gen_dists = []
    min_train_gen_dists = 1
    for i in range(num_lengths):
        num_batches = math.ceil(len(train_seqs[i]) / batch_size) # Do in batches so faster
        for batch in range(num_batches):
            seq_arr = np.array([np.array(s) for s in train_seqs[i][batch * batch_size:(batch + 1) * batch_size]])
            print("batch", batch, "of", num_batches, "seq arr", seq_arr.shape)
            gen_batch = seqs[i * per_length:(i + 1) * per_length]
            all_dist = [batch_hamming(seq_arr, g) for g in gen_batch]
            for list_dist in all_dist:
                if min(list_dist) <= min_train_gen_dists: # Report new min
                    min_train_gen_dists = min(list_dist)
                    print("minimum dis", min_train_gen_dists)
                [train_gen_dists.append(dist) for dist in list_dist if dist <= 0.5]
    all_mins.append(min_train_gen_dists)

[print(runs[i], all_mins[i]) for i in range(len(runs))]
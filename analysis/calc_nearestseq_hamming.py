import numpy as np
import csv
from scipy.spatial import distance
from tqdm import tqdm
from dms.utils import Tokenizer
import math

# run python calc_nearestseq_hamming.py

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

def parse_train_for_length(fasta_file, seq_length):
    """Get all sequences of a certain length from train dataset (used for ESM2 (all 100 res), and FoldingDiff (all diff
    lengths)"""

    seqs = []
    with open(fasta_file, 'r') as file:
        filecontent = csv.reader(file)
        for row in tqdm(filecontent):
            if len(row) >= 1:
                if row[0][0] != '>':
                    if len(row[0]) == seq_length:
                        padded_row = list(row[0])
                        tokenized = [tokenizer.tokenize(s).item() for s in padded_row]
                        seqs.append(tokenized)
    return seqs

def batch_hamming(train_set, g):
    "Compute hamming distance for a batch of data"
    all_dist = [distance.hamming(t, g) for t in train_set]
    return all_dist

project_dir = '../DMs/'
train_fasta = project_dir + 'data/uniref50/' + 'consensus.fasta'
# Calculate each dist to train
runs = ['esm-1b/']
#runs = ['sequence/blosum-0-seq/', 'sequence/oaardm/',
#runs = ['d3pm-final/random-0-seq/', 'arcnn/cnn-38M/', 'pretrain21/cnn-38M/', 'esm-1b/']

batch_size = 10000

# Compute Hamming between all natural sequences
# Uncomment when done getting new data
# train_dists = []
# min_dist = 1
# for i in range(num_lengths):
#     num_batches = math.ceil(len(train_seqs[i])/batch_size)
#     for batch in range(num_batches):
#         seq_arr=np.array([np.array(s) for s in train_seqs[i][batch*batch_size:(batch+1)*batch_size]])
#         print("batch", batch, "of", num_batches, "seq arr", seq_arr.shape)
#         all_dist = list(distance.pdist(np.array(seq_arr), metric='hamming'))
#         if min(all_dist) <= min_dist:
#             min_dist = min(all_dist)
#             print("minimum dis", min_dist)

for run in runs:
    if run =='esm2':
        # For ESM2, they only generated seqs of length 100, so only compare to lengths 100
        per_length = 1000
        num_lengths = 1
        train_seqs = [parse_train_for_length(train_fasta, 100)]
        gen_fasta = project_dir + 'blobfuse/'+run+'generated_samples_string.fasta'
        seqs = tokenize_fasta(gen_fasta)

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

    elif run == 'foldingdiff/': #or run=='esm-1b/': have 1 weird sequence in esm-1b
        print("Gathering seqs len for each sequence")
        # For Folding diff data, find minimum hamming to each len in train of same length
        gen_fasta = project_dir + 'blobfuse/'+run+'generated_samples_string.fasta'
        seqs = tokenize_fasta(gen_fasta)

        train_gen_dists = []
        min_train_gen_dists = 1
        curr_seq_len = 0
        for seq in tqdm(seqs):
            if len(seq) != curr_seq_len:
                train_batch = parse_train_for_length(train_fasta, len(seq))
                curr_seq_len = len(seq)
            all_dist = batch_hamming(train_batch, seq)
            if min(all_dist) <= min_train_gen_dists:
                min_train_gen_dists = min(all_dist)
                print("minimum dist", min_train_gen_dists)

    else: # For everything else we conditionally generated at 4 lengths so can just iterate over train data 1x
        train_seqs = parse_train(train_fasta)  # takes ~ 4 mim
        num_lengths = 4
        per_length = 250
        all_mins = []
        gen_fasta = project_dir + 'blobfuse/' + run + 'generated_samples_string.fasta'
        seqs = tokenize_fasta(gen_fasta)

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
                    #[train_gen_dists.append(dist) for dist in list_dist if dist <= 0.5]
        all_mins.append(min_train_gen_dists)
import numpy as np
import csv
from scipy.spatial import distance
from tqdm import tqdm
from evodiff.utils import Tokenizer
import math
from sequence_models.datasets import UniRefDataset
import pickle
import os

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

# def parse_train(fasta_file):
#     "Get all sequences of the same length from train dataset"
#     seq_lengths = [64, 128, 256, 384]
#     seqs_64 = []
#     seqs_128 = []
#     seqs_256 = []
#     seqs_384 = []
#     with open(fasta_file, 'r') as file:
#         filecontent = csv.reader(file)
#         for row in tqdm(filecontent):
#             if len(row) >= 1:
#                 if row[0][0] != '>':
#                     if len(row[0]) == seq_lengths[0] or len(row[0]) == seq_lengths[1] or len(row[0]) == seq_lengths[2] or len(row[0]) == seq_lengths[3]:
#                         padded_row = list(row[0])
#                         tokenized = [tokenizer.tokenize(s).item() for s in padded_row]
#                         if len(row[0]) == seq_lengths[0]:
#                             seqs_64.append(tokenized)
#                         elif len(row[0]) == seq_lengths[1]:
#                             seqs_128.append(tokenized)
#                         elif len(row[0]) == seq_lengths[2]:
#                             seqs_256.append(tokenized)
#                         elif len(row[0]) == seq_lengths[3]:
#                             seqs_384.append(tokenized)
#     return seqs_64, seqs_128, seqs_256, seqs_384

def parse_train_for_length(fasta_loader):
    from collections import defaultdict
    """Get indices corresponding to sequences at every length and save as pkl file"""
    # Parse entire set once, then use pkl
    if os.path.isfile('train_fasta_lengths.pkl'):
        with open('train_fasta_lengths.pkl', 'rb') as f:
            seqs = pickle.load(f)
    else:
        seqs = defaultdict(list)
        for i in tqdm(range(len(fasta_loader))):
            key = str(len(fasta_loader[i][0]))
            val = i
            seqs[key].append(val)
        with open('train_fasta_lengths.pkl', 'wb') as f:
            pickle.dump(seqs, f)
    return seqs # dict with indices of seqs at var seq lengths

def batch_hamming(train_set, g):
    """Compute hamming distance for a batch of data (train_set) and a given sequence (g)"""
    all_dist = [distance.hamming(t, g) for t in train_set]
    return all_dist

project_dir = ''
train_loader = UniRefDataset('data/uniref50/', 'train', structure=False, max_len=2048)

# Calculate each dist to train
runs = ['d3pm/soar-640M/', 'd3pm_uniform_640M/', 'd3pm_blosum_640M/',
        'hyper12/cnn-650M/', 'esm-1b/',
        'esm2/']
runs = ['arcnn/cnn-38M/', 'sequence/oaardm/', 'd3pm_uniform_38M/', 'd3pm_blosum_38M/',
         'pretrain21/cnn-38M/', 'random-ref/']
compute_natural = False

if compute_natural == True:
    # Compute Hamming between all natural sequences
    train_dists = []
    min_dist = 0.1
    num_lengths= 1
    train_seqs = parse_train(train_fasta)
    for i in range(num_lengths):
        num_batches = math.ceil(len(train_seqs[i])/batch_size)
        for batch in range(num_batches):
            seq_arr=np.array([np.array(s) for s in train_seqs[i][batch*batch_size:(batch+1)*batch_size]])
            print("batch", batch, "of", num_batches, "seq arr", seq_arr.shape)
            all_dist = list(distance.pdist(np.array(seq_arr), metric='hamming'))
            if min(all_dist) <= min_dist:
                print(min(all_dist))
                print([tokenizer.untokenize(seq) for seq in seq_arr])
                min_dist = min(all_dist)
                print("minimum dis", min_dist)

for run in runs:
    print(run)
    all_mins = []
    print("Gathering seqs len for each sequence")
    gen_fasta = project_dir + 'blobfuse/'+run+'generated_samples_string.fasta'
    train_dict = parse_train_for_length(train_loader)
    seqs = tokenize_fasta(gen_fasta)
    train_gen_dists = []
    include_list = [200, 380, 669, 945, 876] # indices to print out hamming for ; (now save all to file - don't need)
    for seq_count, seq in tqdm(enumerate(seqs)):
        train_batch = [[tokenizer.tokenize(s).item() for s in train_loader[idx][0]] for idx in train_dict[str(len(seq))]] # select indices corresponding to dict entry for seq_len
        curr_seq_len = len(seq)
        if len(train_batch)>0:
            all_dist = batch_hamming(train_batch, seq)
        all_mins.append(min(all_dist)) # append minimum hamming per seq
        if seq_count in include_list:
            print("seq", seq_count, min(all_dist))
    # write to file
    out_file = 'blobfuse/'+ run + 'hamming_similarity.csv'
    with open(out_file, 'w') as f:
        [f.write(str(line) + "\n") for line in all_mins]
    f.close()
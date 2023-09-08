from sequence_models.datasets import UniRefDataset
from sequence_models.samplers import SortishSampler, ApproxBatchSampler
import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import numpy as np
import pathlib
from collections import Counter
import csv

# Calculate the AA freq in train, valid, test sets respectively

### SET RANDOM SEEDS ###
random_seed = 1
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)
torch.cuda.empty_cache() # empty caches

dataset='uniref50'
bucket_size = 1000
max_tokens = 40000
max_batch_size = 256

home = str(pathlib.Path.home())
data_top_dir = home + '/Desktop/DMs/data/'
data_dir = data_top_dir + dataset + '/'

metadata = np.load(data_dir + 'lengths_and_offsets.npz')
ds_train = UniRefDataset(data_dir, 'train', structure=False)
ds_valid = UniRefDataset(data_dir, 'rtest', structure=False) # Using test for analysis
valid_idx = ds_valid.indices
train_idx = ds_train.indices
len_train = metadata['ells'][train_idx]
train_sortish_sampler = SortishSampler(len_train, bucket_size, num_replicas=1, rank=0)

train_sampler = ApproxBatchSampler(train_sortish_sampler, max_tokens, max_batch_size, len_train)
valid_sampler = Subset(ds_valid, valid_idx)
dl_train = DataLoader(dataset=ds_train,
                          batch_sampler=train_sampler,
                          num_workers=16)

len_valid = metadata['ells'][valid_idx]
valid_sortish_sampler = SortishSampler(len_valid, 1000, num_replicas=1, rank=0)
valid_sampler = ApproxBatchSampler(valid_sortish_sampler, max_tokens // 2, max_batch_size, len_valid)
dl_valid = DataLoader(dataset=ds_valid,
                  batch_sampler=valid_sampler,
                  num_workers=8)

aminos = Counter({'A':0, 'M':0, 'R':0, 'T':0, 'D':0, 'Y':0, 'P':0, 'F':0, 'L':0, 'E':0, 'W':0, 'I':0, 'N':0, 'S':0,\
                      'K':0, 'Q':0, 'H':0, 'V':0, 'G':0, 'C':0, 'X':0, 'B':0, 'Z':0, 'J':0, 'O':0, 'U':0})
seq_len = []

for i, batch in enumerate(dl_valid):
    for seq in batch[0]:
        aminos.update(seq)
        print(len(seq))
        seq_len.append(len(seq))
    if i % 1000 == 0 :
        print("writing batch ", i)
        with open('count3/batch-count-test'+str(i)+'.csv', 'w') as file:
            writer = csv.DictWriter(file, aminos.keys())
            writer.writeheader()
            writer.writerow(aminos)

        with open('count3/seq_len-test'+str(i)+'.csv', 'w') as file2:
            for item in seq_len:
                # write each item on a new line
                file2.write("%s\n" % item)
            print('Done')

from sequence_models.datasets import UniRefDataset
from sequence_models.samplers import SortishSampler, ApproxBatchSampler
import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import numpy as np
import pathlib
from collections import Counter
import csv

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
train_idx = ds_train.indices

len_train = metadata['ells'][train_idx]
train_sortish_sampler = SortishSampler(len_train, bucket_size, num_replicas=1, rank=0)

#mini_size = 100
#tindices = np.arange(21546293,31546293,1)#(1000000,21546293, 1)
#train_indices = np.sort(np.random.choice(tindices, mini_size, replace=False))
#train_sampler = Subset(ds_train,train_indices)
train_sampler = ApproxBatchSampler(train_sortish_sampler, max_tokens, max_batch_size, len_train)
# dl_train = DataLoader(dataset=train_sampler,
#                               shuffle=True,
#                               batch_size=20,
#                               num_workers=4)
dl_train = DataLoader(dataset=ds_train,
                          batch_sampler=train_sampler,
                          num_workers=16)
# valid_sortish_sampler = SortishSampler(len_train, 80000, num_replicas=1, rank=0)
# valid_sampler = ApproxBatchSampler(valid_sortish_sampler, max_tokens // 2, max_batch_size, len_train)
# dl_train = DataLoader(dataset=ds_train,
#                       batch_sampler=valid_sampler,
#                       num_workers=8)
#letters = Counter({})
aminos = Counter({'A':0, 'M':0, 'R':0, 'T':0, 'D':0, 'Y':0, 'P':0, 'F':0, 'L':0, 'E':0, 'W':0, 'I':0, 'N':0, 'S':0,\
                      'K':0, 'Q':0, 'H':0, 'V':0, 'G':0, 'C':0, 'X':0, 'B':0, 'Z':0, 'J':0, 'O':0, 'U':0})
seq_len = []

for i, batch in enumerate(dl_train):
    for seq in batch[0]:
        aminos.update(seq)
        print(len(seq))
        seq_len.append(len(seq))
    if i % 1000 == 0 :
        print("writing batch ", i)
        with open('count2/batch-count-valid'+str(i)+'.csv', 'w') as file:
            writer = csv.DictWriter(file, aminos.keys())
            writer.writeheader()
            writer.writerow(aminos)

        with open('count2/seq_len-valid'+str(i)+'.csv', 'w') as file2:
            for item in seq_len:
                # write each item on a new line
                file2.write("%s\n" % item)
            print('Done')

from dms.data import loadMatrix
import torch
import numpy as np
from sequence_models.constants import MASK, MSA_PAD, MSA_ALPHABET, MSA_AAS
from dms.constants import BLOSUM_ALPHABET
from sklearn.preprocessing import normalize


def cumprod_matrix(a):
    "takes a list of transition matrices and ouputs a list of the cum prod (Q_bar) at each timestep"
    a_bar = [a[0]]  # initialize w/ first item in list
    start = a[0]
    for i in range(len(a) - 1):
        a_prod_temp = torch.mm(start, a[i + 1])
        start = a_prod_temp
        a_bar.append(a_prod_temp)  # update start
    return a_bar

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def double_stochastic(q):
    q_norm = normalize(q, axis=1, norm='l1')
    while not np.isclose(np.min(np.sum(q_norm, axis=0)), 1): # only checking that one value converges to 1 (prob best to do all 4 min/max)
        q_norm = normalize(q_norm, axis=0, norm='l1')
        q_norm = normalize(q_norm, axis=1, norm='l1')
    return q_norm

def _beta_schedule(num_timesteps, schedule='linear', start=1e-5, end=0.999, max=8):
    """
    Variance schedule for adding noise as introduced by Nichol and Dhariwal and adapted by Hoogeboom et al
    Coined as uniform schedule in Austin et al.
    Start/End will control the magnitude of sigmoidal and cosine schedules..
    """
    if schedule == 'linear':
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == 'sohl-dickstein':
        betas = torch.linspace(0,num_timesteps-1, num_timesteps)
        betas = 1/(num_timesteps - betas + 1)
    elif schedule == "cosine":
        betas = torch.linspace(np.pi / 2, 0, num_timesteps)
        betas = torch.cos(betas) * (end - start) + start
    elif schedule == "exp":
        betas = torch.linspace(0, max, num_timesteps)
        betas = torch.exp(betas) * (end - start) + start
    else:
        print("Must select a valid schedule; ['linear', 'sohl-dickstein', 'cosine', 'exp']")
    return betas

def read_fasta(fasta_path, seq_file, info_file, index_file):
    """
    Read fasta and extract sequences, write out a corresponding index file w/ headers
    Only needs to be done 1x to clean data
    """
    with open(fasta_path) as f_in, open(seq_file, 'w') as f_out, open(info_file, 'w') as info_out, open(index_file, 'w') as i_out:
        current_seq = ''  # sequence string
        index = 0
        for line in f_in:
            if line[0] == '>':
                # print(line)
                i_out.write(str(index)+"\n")
                info_out.write(line)  # line containing seq info
                index+=1
                current_seq += "\n"
                # print(len(current_seq))
                f_out.write(current_seq)
                current_seq = ''  # new line for new seq
            else:
                current_seq += line[:-1]

def parse_fasta(seq_file, idx):
    """
    Reads seq_file from processing steps, and will extract sequence at a given index
    """
    sequence = ''
    with open(seq_file) as f_in:
        for l, line in enumerate(f_in):
            if l == idx:
                sequence += (line[:-1])
                break
    return sequence


class Tokenizer(object):
    """Convert between strings and index"""
    def __init__(self, protein_alphabet=MSA_ALPHABET, pad=MSA_PAD, mask=MASK, all_aas=MSA_AAS, path_to_blosum=None):
        self.alphabet = list("".join(protein_alphabet))
        self.all_aas = list("".join(all_aas))
        self.pad = pad
        self.mask = mask
        self.a_to_i = {u: i for i, u in enumerate(self.alphabet)}
        self.i_to_a = np.array(self.alphabet)
        if path_to_blosum is not None:
            self.matrix = loadMatrix(path_to_blosum)
            self.matrix_dict = dict(self.matrix)

    @property
    def pad_id(self):
         return self.tokenize(self.pad)[0]

    @property
    def mask_id(self):
        return self.tokenize(self.mask)[0]

    def q_blosum(self):
        K = len(self.all_aas)
        q = np.array([i for i in self.matrix_dict.values()])
        q = q.reshape((K,K))
        q = softmax(q)
        q = double_stochastic(q)
        q = torch.tensor(q)
        #print(q.sum(axis=0), q.sum(axis=1))
        #EXPAND DIMENSIONS TO MATCH MSA_ALPHABET
        # P = len(self.alphabet)
        # q_expand = torch.zeros(P, P)
        # q_i, q_j = q.shape
        # for i, row in enumerate(q_expand):
        #     for j, value in enumerate(row):
        #         # columns
        #         if (i <= q_i - 1) and (j <= q_j - 1):
        #             q_expand[i, j] = q[i, j]
        #         else: # fill anything not in matrix with zeros
        #             q_expand[i, j] = 0.0
        # REORDER BLOSUM MATRIX BASED ON MSA_ALPHABET (self.alphabet, self.a_to_i)
        new_q = q.clone()
        i2_to_a = np.array(list(BLOSUM_ALPHABET))
        for i, row in enumerate(new_q):
            for j, value in enumerate(row):
                ind1, ind2 = [i, j]
                key = i2_to_a[ind1], i2_to_a[ind2]
                new1, new2 = [self.a_to_i[k] for k in key]
                #print([ind1, ind2], key, [new1, new2])
                #print("before", new_q[new1,new2])
                new_q[new1, new2] = q[ind1, ind2]
                #print("after", new_q[new1,new2])
        #print(new_q == q)
        #print(new_q.sum(axis=0), new_q.sum(axis=1))
        return new_q

    def q_blosum_schedule(self, timesteps=500, schedule='exp', max=6):
        """
        betas = None; Natural mutation pattern for blosum - no schedule
        betas = 'exp' use exp scheme for beta schedule
        """
        print(schedule)
        K = len(self.all_aas)
        #print(self.alphabet)
        q = self.q_blosum()
        _betas = _beta_schedule(timesteps, schedule=schedule, max=max)
        betas = (_betas - _betas.min())
        betas = betas / betas.max()
        Q_t = [] # scheduled matrix
        for i in range(timesteps):
            q_non_diag = torch.ones((K,K)) * q * betas[i]
            norm_constant = (1 - (q_non_diag).sum(axis=0))
            q_diag = torch.tensor(np.identity(K)) * norm_constant
            R = q_diag + q_non_diag
            Q_t.append(R)
        Q_prod = cumprod_matrix(Q_t)
        Q_prod = torch.stack(Q_prod) # cumprod of matrices
        Q_t = torch.stack(Q_t) # scheduled matrix
        return Q_prod, Q_t

    def q_random_schedule(self, timesteps=500, schedule='sohl-dickstein'):
        print(schedule)
        betas = _beta_schedule(timesteps, schedule=schedule)
        betas = betas / betas.max()
        K = len(self.all_aas)
        Q_t = []  # scheduled matrix
        for i in range(len(betas)):
            q_non_diag = torch.ones((K,K)) / K * betas[i]
            norm_constant = (1 - (q_non_diag).sum(axis=0))
            q_diag = torch.tensor(np.identity(K)) * norm_constant
            R = q_diag + q_non_diag
            Q_t.append(R)
        Q_prod = cumprod_matrix(Q_t)
        Q_prod = torch.stack(Q_prod)  # cumprod of matrices
        Q_t = torch.stack(Q_t)  # scheduled matrix
        return Q_prod, Q_t

    def tokenize(self, seq):
        return np.array([self.a_to_i[a] for a in seq[0]]) # for nested lists

    def tokenizeMSA(self, seq):
        return np.array([self.a_to_i[a] for a in seq]) # not nested

    def untokenize(self, x):
        if torch.is_tensor(x):
            return "".join([self.i_to_a[int(t.item())] for t in x])
        else:
            return "".join([self.i_to_a[t] for t in x])

    def one_hot(self, tokenized):
        "one hot encode according to indexing"
        x_onehot = torch.nn.functional.one_hot(tokenized, num_classes=len(self.all_aas))
        return x_onehot.to(torch.double)

    def undo_one_hot(self, x_onehot):
        "one hot -> seq"
        tokenized = [np.where(r==1)[0] for r in x_onehot] # TODO may need to fix now that using torch nn have not double checked
        return tokenized

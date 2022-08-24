from dms.data import loadMatrix
import torch
import numpy as np
from sequence_models.constants import MASK
from dms.constants import ALL_AAS, PROTEIN_ALPHABET, PAD
from sklearn.preprocessing import normalize

def matrixMul(a, n):
    if(n <= 1):
        return a
    else:
        return torch.matmul(matrixMul(a, n-1), a)

def matrixProd(a):
    "Takes a matrix of size T total timesteps as input and calculates the cumulative product at each t step"
    matrix = torch.zeros(a.shape[0], a.shape[1], a.shape[2])
    start = a[0]
    for i in range(len(a)):
        start = torch.matmul(start, a[i])
        matrix[i] = start
    return matrix.to(torch.double)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def double_stochastic(q):
    q_norm = normalize(q, axis=1, norm='l1')
    while not np.isclose(np.min(np.sum(q_norm, axis=0)), 1): # only checking that one value converges to 1 (prob best to do all 4 min/max)
        q_norm = normalize(q_norm, axis=0, norm='l1')
        q_norm = normalize(q_norm, axis=1, norm='l1')
    return q_norm

# def _beta_schedule(timesteps, schedule='linear'):
#     if schedule == 'sohl-dickstein':
#         betas = torch.linspace(0,timesteps-1, timesteps)
#         betas = 1/(timesteps - betas + 1)
#         alphas = 1-betas
#     elif schedule == "cosine":
#         betas = torch.linspace(0, np.pi, timesteps)
#         betas = torch.cos(betas)
#         betas = (betas - betas.min())
#         betas = betas/betas.max()
#         alphas = 1 - betas
#         #betas = betas-betas.min()
#         #alphas = betas.max()-betas
#     elif schedule == 'linear':
#         betas = torch.linspace(0.02, 1, timesteps)
#         alphas = 1-betas
#     return betas, alphas

def _beta_schedule(num_timesteps, schedule='linear', start=1e-5, end=0.999, max=8, timesteps=500):
    """
    Variance schedule for adding noise as introduced by Nichol and Dhariwal and adapted by Hoogeboom et al
    Coined as uniform schedule in Austin et al.
    Start/End will control the magnitude of sigmoidal and cosine schedules..
    #TODO: Check that cosine matches Austin cosine schedule - I think theirs is slightly diff
    #TODO: add mutual information Beta_t introduced by Sohl Dickensen used by Austin
    """
    if schedule == 'linear':
        betas = torch.linspace(start, end, num_timesteps)
    if schedule == 'sohl-dickstein':
        betas = torch.linspace(0,timesteps-1, timesteps)
        betas = 1/(timesteps - betas + 1)
    elif schedule == "cosine":
        betas = torch.linspace(np.pi / 2, 0, num_timesteps)
        betas = torch.cos(betas) * (end - start) + start
    elif schedule == "sine":
        betas = torch.linspace(np.pi/2, 0, num_timesteps)
        betas = torch.sin(betas) * (end - start) + start
    elif schedule == "exp":
        betas = torch.linspace(0, max, num_timesteps)
        betas = torch.exp(betas) * (end - start) + start
    else:
        print("Must select a valid schedule; ['linear', 'quad', 'sigmoid', 'cosine']")
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
    def __init__(self, all_aas=ALL_AAS, protein_alphabet=PROTEIN_ALPHABET, pad=PAD, mask=MASK, path_to_blosum=None, masking_scheme='blosum'):
        self.all_aas = list(all_aas)
        self.alphabet = list("".join(protein_alphabet))
        self.pad = pad
        self.mask = mask
        self.vocab = sorted(set("".join(all_aas)))
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
        q = np.array([i for i in self.matrix_dict.values()])
        q = q.reshape((len(self.all_aas), len(self.all_aas)))
        q = softmax(q)
        q = double_stochastic(q)
        return q

# THIS ONE WORKS BEST FOR SOME REASON ALTHOUGH IT SHOULDNT AND IS WRONG ###
# ATTEMPT 2 -> 0.5 ACC IN 500 EPOCHS
#     def q_blosum_schedule(self, timesteps=500):
#         q = torch.tensor(self.q_blosum())
#         betas = _beta_schedule(timesteps, 'cosine')
#         betas = betas / betas.max()  # normalize first value to 0
#         alphas = 1 - betas
#         q_diag = torch.tensor(np.identity(len(self.all_aas))) * q
#         q_non_diag = torch.tensor((1 - np.identity(len(self.all_aas)))) * q
#         q_t = []
#         for i, a in enumerate(alphas):
#             b = betas[i]
#             R = q_diag * a + q_non_diag * b
#             q_temp = R
#             q_t.append(q_temp)
#         q_t = torch.stack(q_t)
#         return q_t

    def q_blosum_schedule(self, timesteps=500, betas=None):
        """
        betas = None; Natural mutation pattern for blosum - no schedule
        betas = 'exp' use exp scheme for beta schedule
        """
        K = len(self.all_aas)
        q = torch.tensor(self.q_blosum())
        if betas is not None:
            _betas = _beta_schedule(timesteps, 'exp', max=6)
            betas = (_betas - _betas.min())
            betas = betas / betas.max()
        else:
            _betas = torch.ones(timesteps)
        q_t = []
        for i in range(timesteps):
            q_non_diag = torch.tensor((1 - np.identity(K))) * q * betas[i]
            norm_constant = (1 - (q_non_diag).sum(axis=0))
            q_diag = torch.tensor(np.identity(K)) * norm_constant
            R = q_diag + q_non_diag
            q_temp = matrixMul(R, i)
            q_t.append(q_temp)
        q_t = torch.stack(q_t)
        return q_t

    def q_random_schedule(self, timesteps=500):
        betas = _beta_schedule(timesteps, 'cosine')
        K = len(self.all_aas)
        q_t = []
        for i in range(len(betas)):
            q_non_diag = torch.tensor((1 - np.identity(K))) / K * betas[i]
            norm_constant = (1 - (q_non_diag).sum(axis=0))
            q_diag = torch.tensor(np.identity(K)) * norm_constant
            R = q_diag + q_non_diag
            #print(R.sum(axis=0), R.sum(axis=1))
            q_t.append(R)
        q_t = torch.stack(q_t)
        return q_t

    def tokenize(self, seq):
        return np.array([self.a_to_i[a] for a in seq[0]]) # seq is a tuple with empty second dim

    def untokenize(self, x):
        if torch.is_tensor(x):
            return "".join([self.i_to_a[int(t.item())] for t in x])
        else:
            return "".join([self.i_to_a[t] for t in x])

    def one_hot(self, seq, tokenized=False):
        "one hot encode according to indexing"
        tokens = self.all_aas
        x_onehot = np.zeros((len(seq), len(tokens)))
        for i, a in enumerate(seq):
            if not tokenized:
                one_index = self.a_to_i[a]
            else:
                one_index = a
            if one_index < len(tokens): # everything that isnt an amino acid will be zero
                x_onehot[i][int(one_index)] = 1
        return x_onehot

    def undo_one_hot(self, x_onehot):
        "one hot encode according to indexing"
        tokenized = [np.where(r==1)[0] for r in x_onehot]
        return tokenized

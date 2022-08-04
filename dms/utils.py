#import blosum as bl
from dms.data import loadMatrix
import torch
import numpy as np
from sequence_models.constants import MASK
from dms.constants import ALL_AAS, PROTEIN_ALPHABET, PAD
from sklearn.preprocessing import normalize

data_dir = '/home/v-salamdari/Desktop/DMs/data/' # TODO fix this

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def double_stochastic(q):
    q_norm = normalize(q, axis=1, norm='l1')
    while not np.isclose(np.min(np.sum(q_norm, axis=0)), 1): # only checking that one value converges to 1 (prob best to do all 4 min/max)
        q_norm = normalize(q_norm, axis=0, norm='l1')
        q_norm = normalize(q_norm, axis=1, norm='l1')
    return q_norm

def _beta_schedule(num_timesteps, schedule='linear', start=1e-5, end=0.999):
    """
    Variance schedule for adding noise as introduced by Nichol and Dhariwal and adapted by Hoogeboom et al
    Coined as uniform schedule in Austin et al.
    Start/End will control the magnitude of sigmoidal and cosine schedules..
    #TODO: Check that cosine matches Austin cosine schedule - I think theirs is slightly diff
    #TODO: add mutual information Beta_t introduced by Sohl Dickensen used by Austin
    """
    if schedule == 'linear':
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-10, 10, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine":
        betas = torch.linspace(np.pi / 2, 0, num_timesteps)
        betas = torch.cos(betas) * (end - start) + start
    elif schedule == "sine":
        betas = torch.linspace(np.pi / 2, 0, num_timesteps)
        betas = torch.sin(betas) * (end - start) + start
    elif schedule == "exp":
        betas = torch.linspace(0, 8, num_timesteps) # TOOO: this 8 should be a tuneable parameter
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
    def __init__(self, all_aas=ALL_AAS, protein_alphabet=PROTEIN_ALPHABET, pad=PAD, mask=MASK,
                 path_to_blosum=data_dir+"blosum62-special.mat"):
        self.matrix = loadMatrix(path_to_blosum)
        self.matrix_dict = dict(self.matrix)
        self.all_aas = list(all_aas)
        self.alphabet = list("".join(protein_alphabet))
        self.pad = pad
        self.mask = mask
        self.vocab = sorted(set("".join(all_aas)))
        self.a_to_i = {u: i for i, u in enumerate(self.alphabet)}
        self.i_to_a = np.array(self.alphabet)

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
        return torch.tensor(q)

    def q_blosum_schedule(self, timesteps=500, end=0.4):
        q = self.q_blosum()
        betas = _beta_schedule(timesteps, 'exp', end=end)
        alphas = betas - end # normalize first value to 0
        q_diag = torch.tensor(np.identity(len(self.all_aas))) * q
        q_non_diag = torch.tensor((1 - np.identity(len(self.all_aas)))) * q
        #print(q_diag, q_non_diag)
        q_t = []
        for i, a in enumerate(alphas):
            R = q_diag + q_non_diag * a
            q_temp = double_stochastic(R)
            q_t.append(torch.tensor(q_temp))
        q_t = torch.stack(q_t)
        return q_t

    def q_random(self):
        q = np.eye(len(self.alphabet)) + 1 / 10  # arbitrary, set diag to zero assign other transitions some prob
        q = double_stochastic(q)  # normalize so rows += 1
        return q

    def q_blosum_scaled(self, alpha_t=0.03):
        q = self.q_blosum
        q_diag = np.identity(len(self.all_aas)) * q
        q_non_diag = (1 - np.identity(len(self.all_aas))) * q
        q_alpha_t = double_stochastic((q_diag + np.dot(q_non_diag, np.array(alpha_t))))
        return q_alpha_t

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
            #print(one_index, len(tokens))
            if one_index < len(tokens): # everything that isnt an amino acid will be zero
                #print(i, one_index)
                x_onehot[i][int(one_index)] = 1
        return x_onehot

    def undo_one_hot(self, x_onehot):
        "one hot encode according to indexing"
        tokenized = [np.where(r==1)[0] for r in x_onehot]
        return tokenized

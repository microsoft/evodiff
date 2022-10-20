import numpy as np
import torch
from dms.utils import Tokenizer


def _pad(tokenized, value, dim=2):
    """
    Utility function that pads batches to the same length.

    tokenized: list of tokenized sequences
    value: pad index
    """
    batch_size = len(tokenized)
    max_len = max(len(t) for t in tokenized)
    if dim == 3: # dim = 3 (one hot)
        categories = tokenized[0].shape[-1]
        output = torch.zeros((batch_size, max_len, categories)) + value
        for row, t in enumerate(tokenized):
            output[row, :len(t), :] = t
    elif dim == 2: # dim = 2 (tokenized)
        output = torch.zeros((batch_size, max_len)) + value
        for row, t in enumerate(tokenized):
            output[row, :len(t)] = t
    else:
        print("padding not supported for dim > 3")
    return output

def _pad_msa(tokenized, num_seq, max_len, value, dim=3):
    """Utility function that pads batches to the same length."""
    batch_size = len(tokenized)
    if dim == 4: # one hot MSA
        categories = tokenized[0].shape[-1] # last dim is one hot
        output = torch.zeros((batch_size, num_seq, max_len, categories), dtype=torch.long) + value
        for i in range(batch_size):
            output[i, :, :len(tokenized[i][0]), :] = tokenized[i]
    elif dim == 3: # tokenized MSA
        output = torch.zeros((batch_size, num_seq, max_len), dtype=torch.long) + value
        for i in range(batch_size):
            output[i, :, :len(tokenized[i][0])] = tokenized[i]
    else:
        print("padding not supported for dim > 4")
    return output

def _unpad(x, value):
    x_pad = x.clone()
    mask_pad = x_pad != value
    x = x[mask_pad].to(torch.int64)
    return x

def sample_transition_matrix(x_0, Q_bar):
    """
    Sample a markov transition according to next_step = x_0 * q ^ time,
    where Q_bar = q ^t or cumprod of scheduled transition matrices
    returns sample and probabilities
    """
    p_next_step = torch.mm(x_0, Q_bar)
    next_step = torch.multinomial(p_next_step, num_samples=1)
    return next_step.squeeze(), p_next_step # sample and probabilities


class OAMaskCollater(object):
    """
    OrderAgnosic Mask Collater for masking batch data according to Hoogeboom et al. OA ARDMS
    inputs:
        sequences : list of sequences
        inputs_padded: if inputs are padded (due to truncation in Simple_Collater) set True (default False)

    OA-ARM variables:
        D : possible permutations from 0.. max length
        t : randomly selected timestep

    outputs:
        src : source  masked sequences (model input)
        timesteps: (D-t+1) term
        tokenized: tokenized sequences (target seq)
        masks: masks used to generate src
    """
    def __init__(self, tokenizer=Tokenizer()):
        self.tokenizer = tokenizer

    def __call__(self, sequences):
        tokenized = [torch.tensor(self.tokenizer.tokenize(s)) for s in sequences]
        max_len = max(len(t) for t in tokenized)
        src=[]
        timesteps = []
        masks=[]
        mask_id = torch.tensor(self.tokenizer.mask_id, dtype=torch.int64)
        for i,x in enumerate(tokenized):
            # Randomly generate timestep and indices to mask
            D = len(x) # D should have the same dimensions as each sequence length
            if D <= 1:  # for sequence length = 1 in dataset
                t = 1
            else:
                t = np.random.randint(1, D) # randomly sample timestep
            num_mask = (D-t+1) # from OA-ARMS
            # Append timestep
            timesteps.append(num_mask)
            # Generate mask
            mask_arr = np.random.choice(D, num_mask, replace=False) # Generates array of len num_mask
            index_arr = np.arange(0, max_len) #index array [1...seq_len]
            mask = np.isin(index_arr, mask_arr, invert=False).reshape(index_arr.shape) # mask bools indices specified by mask_arr
            # Mask inputs
            mask = torch.tensor(mask, dtype=torch.bool)
            masks.append(mask)
            x_t = ~mask[0:D] * x + mask[0:D] * mask_id
            src.append(x_t)
        # PAD out
        src = _pad(src, self.tokenizer.pad_id)
        masks = _pad(masks*1,0) #, self.seq_length, 0)
        tokenized = _pad(tokenized, self.tokenizer.pad_id)
        return (src.to(torch.long), torch.tensor(timesteps), tokenized.to(torch.long), masks)

class D3PMCollater(object):
    """
    D3PM Collater for generating batch data according to markov process according to Austin et al.
    inputs:
        sequences : list of sequences
        tokenizer: Tokenizer()
        masking scheme: 'BLOSUM' uses blosum matrix, 'RANDOM' uses uniform transition matrix
        num_timesteps: number of diffusion timesteps

    outputs:
        src : source  masked sequences (model input)
        timesteps: (D-t+1) term
        tokenized: tokenized sequences (target seq)
        masks: masks used to generate src
        Q : markov matrix
        q_x : forward transition probabilities
    """
    def __init__(self, tokenizer=Tokenizer(), num_timesteps=100, Q=None, Q_bar=None):
        self.tokenizer = tokenizer
        self.num_timesteps = num_timesteps # Only needed for markov trans, doesnt depend on seq len
        self.alphabet = self.tokenizer.tokenize([self.tokenizer.alphabet])
        self.all_aas = self.tokenizer.tokenize([self.tokenizer.all_aas])
        self.Q = Q
        self.Q_bar =Q_bar

    def __call__(self, sequences):
        tokenized = [torch.tensor(self.tokenizer.tokenize(s)) for s in sequences]
        one_hot = []
        ## This is to deal with an empty sequence ##
        del_index = None
        for i,t in enumerate(tokenized):
            if len(t) == 0: # TODO: was this an old bug? can i delete now -check ignore empty sequence in MNIST dataset
                one_hot.append(torch.zeros(len(self.all_aas), dtype=torch.double))
                del_index = i
            else:
                one_hot.append(self.tokenizer.one_hot(t))
        if del_index is not None:
            tokenized.pop(del_index)
            one_hot.pop(del_index)
        max_len = max(len(t) for t in tokenized)
        src=[]
        timesteps = []
        # Pre pad one-hot arrays
        pad_one_hot = torch.zeros((len(self.all_aas)))
        q_x = pad_one_hot.repeat((len(tokenized), max_len, 1))
        q_x_minus1 = pad_one_hot.repeat((len(tokenized), max_len, 1))
        for i,x in enumerate(one_hot): # enumerate over batch
            D = len(x) # sequence length
            t = np.random.randint(1, self.num_timesteps) # randomly sample timestep
            # Append timestep
            timesteps.append(t)
            # Calculate forward at time t and t-1
            x_t, q_x_t = sample_transition_matrix(x, self.Q_bar[t]) # x = tgt, x_t = src, Q_bar[t] is cum prod @ time t
            x_tminus1, q_x_tminus1 = sample_transition_matrix(x, self.Q_bar[t-1])
            src.append(x_t)
            q_x[i, :D, :] = q_x_t
            q_x_minus1[i, :D, :] = q_x_tminus1
        # PAD out
        src = _pad(src, self.tokenizer.pad_id)
        tokenized = _pad(tokenized, self.tokenizer.pad_id)
        return (src.to(torch.long), torch.tensor(timesteps), tokenized.to(torch.long),
                self.Q,  q_x.to(torch.double), q_x_minus1.to(torch.double))

class D3PMCollaterMSA(object):
    """
    D3PM Collater for MSAs
    """
    def __init__(self, tokenizer=Tokenizer(), num_timesteps=100,  Q=None, Q_bar=None, num_seqs=64):
        self.tokenizer = tokenizer
        self.num_timesteps = num_timesteps # Only needed for markov trans, doesnt depend on seq len
        self.alphabet = self.tokenizer.tokenize([self.tokenizer.alphabet])
        self.all_aas = self.tokenizer.tokenize([self.tokenizer.all_aas])
        self.Q = Q
        self.Q_bar = Q_bar
        self.num_seqs = num_seqs

    def __call__(self, msas):
        batch_size = len(msas)
        tokenized = list(msas) # tgt
        one_hot = tokenized.copy()
        src = tokenized.copy()
        src_one_hot = tokenized.copy()
        max_seq_len = max(len(t[0]) for t in tokenized) # all seqs in MSA are the same len
        timesteps = []
        # Pre pad one-hot arrays
        pad_one_hot = torch.zeros((len(self.all_aas)))
        q_x = pad_one_hot.repeat((batch_size, self.num_seqs, max_seq_len, 1))
        q_x_minus1 = pad_one_hot.repeat((batch_size, self.num_seqs, max_seq_len, 1))
        for i in range(batch_size): # enumerate over batch
            tokenized[i] = [torch.tensor(self.tokenizer.tokenizeMSA(s)) for s in msas[i]]
            one_hot[i] = [self.tokenizer.one_hot(t) for t in tokenized[i]]
            curr_msa = torch.stack(one_hot[i])
            length, depth, tokens = curr_msa.shape  # length = number of seqs in MSA, depth = # AA in MSA
            curr_msa = curr_msa.flatten(start_dim=0, end_dim=1)
            # Append timestep
            t = np.random.randint(1, self.num_timesteps+1) # randomly sample timestep
            timesteps.append(t)
            # Calculate target
            x_t, q_x_t = sample_transition_matrix(curr_msa, self.Q_bar[t])  # x = tgt, x_t = src, Q_bar accounts for time
            x_tminus1, q_x_tminus1 = sample_transition_matrix(curr_msa, self.Q_bar[t-1])  # x = tgt, x_t = src, Q_bar accounts for time
            x_t = x_t.reshape(length, depth)
            q_x_t = q_x_t.reshape(length, depth, tokens)
            q_x_tminus1 = q_x_tminus1.reshape(length, depth, tokens)
            src[i] = x_t
            src_one_hot[i] = [self.tokenizer.one_hot(t) for t in x_t]
            q_x[i, :, :depth, :] = q_x_t
            q_x_minus1[i, :, :depth, :] = q_x_tminus1
            tokenized[i] = torch.stack(tokenized[i]) # replace list with stack
            one_hot[i] = torch.stack(one_hot[i])
            src_one_hot[i] = torch.stack(src_one_hot[i])
        # PAD out
        src = _pad_msa(src, self.num_seqs, max_seq_len, self.tokenizer.pad_id)
        tokenized = _pad_msa(tokenized, self.num_seqs, max_seq_len, self.tokenizer.pad_id)
        src_one_hot = _pad_msa(src_one_hot, self.num_seqs, max_seq_len, self.tokenizer.pad_id, dim=4)
        return (src.to(torch.long), src_one_hot.to(torch.double), torch.tensor(timesteps), tokenized.to(torch.long),
                self.Q,  q_x.to(torch.double), q_x_minus1.to(torch.double))
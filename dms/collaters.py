import numpy as np
import torch
from dms.utils import Tokenizer, matrixMul
from dms.constants import ALL_AAS

def _pad(tokenized, value, dim=2):
    """
    Utility function that pads batches to the same length.

    tokenized: list of tokenized sequences
    value: pad index
    """
    batch_size = len(tokenized)
    max_len = max(len(t) for t in tokenized)
    if dim > 3:
        "Print dimensions too large to pad"
    elif dim == 3: # dim = 3 (one hot)
        categories = tokenized[0].shape[1]
        output = torch.zeros((batch_size, max_len, categories)) + value
        for row, t in enumerate(tokenized):
            output[row, :len(t), :] = t
    else: # dim = 2 (tokenized)
        output = torch.zeros((batch_size, max_len)) + value
        for row, t in enumerate(tokenized):
            output[row, :len(t)] = t
    return output

def _unpad(x, value):
    x_pad = x.clone()
    mask_pad = x_pad != value
    x = x[mask_pad].to(torch.int64)
    return x

def sample_transition_matrix(x_0, Q, time):
    "Sample a markov transition according to next_step = x_0 * q ^ time"
    p_next_step = torch.matmul(x_0, matrixMul(Q, time))
    next_step = torch.multinomial(p_next_step, num_samples=1)
    return next_step.squeeze(), p_next_step


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
            if D <= 1:  # dataset has sequences length = 1, probably should filter these out
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
    def __init__(self, tokenizer=Tokenizer(), num_timesteps=100, transition_matrix=None):
        self.tokenizer = tokenizer
        self.num_timesteps = num_timesteps # Only needed for markov trans, doesnt depend on seq len
        self.alphabet = self.tokenizer.tokenize([self.tokenizer.alphabet])
        self.all_aas = self.tokenizer.tokenize([self.tokenizer.all_aas])
        self.Q = transition_matrix

    def __call__(self, sequences):
        tokenized = [torch.tensor(self.tokenizer.tokenize(s)) for s in sequences]
        one_hot = [torch.tensor(self.tokenizer.one_hot(s[0])) for s in sequences]
        max_len = max(len(t) for t in tokenized)
        src=[]
        timesteps = []
        masks=[]
        # Pre pad one-hot arrays
        pad_one_hot = torch.zeros((len(self.all_aas)))
        q_x = pad_one_hot.repeat((len(tokenized), max_len, 1))
        for i,x in enumerate(one_hot): # enumerate over batch
            D = len(x) # sequence length
            t = np.random.randint(1, self.num_timesteps) # randomly sample timestep
            # Append timestep
            timesteps.append(t)
            # Calculate target
            x_t, q_x_t = sample_transition_matrix(x, self.Q[t], 1) # x = tgt, x_t = src, Q is already a function of time here
            #print(q_x_t.shape)
            src.append(x_t)
            q_x[i, :D, :] = q_x_t
            # mask = determines which tokens were mutated
            mask = torch.ne(tokenized[i], x_t)
            masks.append(mask)
        # PAD out
        src = _pad(src, self.tokenizer.pad_id)
        masks = _pad(masks*1, 0)
        tokenized = _pad(tokenized, self.tokenizer.pad_id)
        one_hot = _pad(one_hot, self.tokenizer.pad_id, dim=3)
        return (src.to(torch.long), torch.tensor(timesteps), tokenized.to(torch.long), one_hot.to(torch.float16), masks.to(torch.long), self.Q, q_x.to(torch.double))
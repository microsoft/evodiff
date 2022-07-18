import numpy as np
import torch
from dms.utils import Tokenizer, Blosum62
#from sequence_models.constants import STOP
from dms.constants import BLOSUM62_ALPHABET

def _pad(tokenized, value):
    """
    Utility function that pads batches to the same length.

    tokenized: list of tokenized sequences
    value: pad index
    """
    batch_size = len(tokenized)
    max_len = max(len(t) for t in tokenized)
    output = torch.zeros((batch_size, max_len)) + value
    for row, t in enumerate(tokenized):
        #print(t.shape)
        output[row, :len(t)] = t
    return output

def _unpad(x, value):
    x_pad = x.clone()
    mask_pad = x_pad != value
    x = x[mask_pad].to(torch.int64)
    return x

def _normalize_seq_lengths(tokenized, seq_length, value):
    """
    Utility function that pads batches to the same length.
    Will always pad to right of sequence

    tokenized : list of tokenized sequences
    seq_length: (int) length to normalize sequences to
    value: pad index
    """
    batch_size = len(tokenized)
    output = torch.zeros((batch_size, seq_length)) + value
    for row, t in enumerate(tokenized):
        if len(t) <= seq_length:
            output[row, :len(t)] = t
        else:
            output[row, :seq_length] = t[:seq_length]
    return output

def matrixMul(a, n):
    if(n <= 1):
        return a
    else:
        return np.matmul(matrixMul(a, n-1), a)

def random_sample(seq, p, alphabet):
    sampled_seq = torch.zeros(len(seq))
    #print(len(alphabet), alphabet)
    for i in range(len(seq)):
        aa_selected = np.random.choice(len(alphabet), p=p[i])
        sampled_seq[i] = aa_selected
    return sampled_seq

def sample_transition_matrix(x_0, q, time, alphabet):
    "Sample a markov transition according to next_step = x_0 * q ^ time"
    p_next_step = np.matmul(x_0, matrixMul(q, time))
    next_step = random_sample(x_0, p_next_step, alphabet)
    return next_step

def _diff(a, b):
    return [i for i in range(len(a)) if a[i] != b[i]]

# def _beta_schedule(num_timesteps, schedule='linear', start=1e-5, end=0.999):
#     """
#     Variance schedule for adding noise as introduced by Nichol and Dhariwal and adapted by Hoogeboom et al
#     Coined as uniform schedule in Austin et al.
#     Start/End will control the magnitude of sigmoidal and cosine schedules..
#     #TODO: Check that cosine matches Austin cosine schedule - I think theirs is slightly diff
#     #TODO: add mutual information Beta_t introduced by Sohl Dickensen used by Austin
#     """
#     if schedule == 'linear':
#         betas = torch.linspace(start, end, num_timesteps)
#     elif schedule == "quad":
#         betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
#     elif schedule == "sigmoid":
#         betas = torch.linspace(-10, 10, num_timesteps)
#         betas = torch.sigmoid(betas) * (end - start) + start
#     elif schedule == "cosine":
#         betas = torch.linspace(np.pi / 2, 0, num_timesteps)
#         betas = torch.cos(betas) * (end - start) + start
#     else:
#         print("Must select a valid schedule; ['linear', 'quad', 'sigmoid', 'cosine']")
#     return betas

# def blosum_probability(tokenized):
#     d = Blosum62().blosum_dict()
#     out = torch.zeros((len(tokenized)))
#     for i, x in enumerate(tokenized):
#         if i < len(tokenized) - 1:
#             _tuple = tuple(np.array(tokenized[i:i + 2]))
#         else:
#             end = tokenized[PROTEIN_ALPHABET.index(STOP)]
#             _tuple = tuple(np.array((tokenized[i], end)))
#         out[i] = d[_tuple]
#     return out


class SimpleCollater(object):
    """
    Slightly altered from protein-sequence-models (K. Yang)

    Performs simple operations on batch of sequences contained in a list
    - Can reverse sequence orders
    - Pad to "max" length in batch or pads to "normalized" length chosen by providing seq_length (default 512)
    """

    def __init__(self, pad=False, backwards=False, norm=False, seq_length=512, one_hot=False):
        self.pad = pad
        self.seq_length = seq_length
        self.tokenizer = Tokenizer()
        self.blosum = Blosum62()  #Blosum62 for one_hot
        self.backwards = backwards
        self.norm = norm
        self.one_hot=one_hot

    def __call__(self, batch):
        prepped = self._prep(batch)
        return prepped

    def _prep(self, sequences):
        if self.backwards:
            sequences = [s[::-1] for s in sequences]

        if self.one_hot:
             #print([len(s[0]) for s in sequences])
             tokenized = [torch.LongTensor(self.blosum.one_hot(s[0])) for s in sequences]
        else:
            tokenized = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in sequences]
            #print("called here2")

        if self.norm:
            #print("Truncating sequences to ", self.seq_length, "and padding")
            tokenized = _normalize_seq_lengths(tokenized, self.seq_length, self.tokenizer.pad_id)
            #print("called here3")

        if self.pad:
            #print("Padding sequences to max seq length")
            tokenized = _pad(tokenized, self.tokenizer.pad_id)
            #print("called here4")

        #print("Tokenizing sequences")
        return tokenized

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
    def __init__(self, simple_collater, inputs_padded=False):
        self.simple_collater = simple_collater
        self.tokenizer = Tokenizer()
        self.inputs_padded  = inputs_padded

    def __call__(self, sequences):
        tokenized = self.simple_collater(sequences)
        max_len = max(len(t) for t in tokenized)
        #print("max len", max_len)
        src=[]
        timesteps = []
        masks=[]
        mask_id = torch.tensor(self.tokenizer.mask_id, dtype=torch.int64)

        #batch_size = len(tokenized)

        for i,x in enumerate(tokenized):
            if self.inputs_padded: # if truncating seqs to some length first in SimpleCollater, inputs will be padded
                 x_pad = x.clone()
                 mask_pad = x_pad != self.tokenizer.pad_id
                 #num_pad = len(x_pad) - mask_pad.sum()
                 x = x[mask_pad].to(torch.int64)
            # Randomly generate timestep and indices to mask
            D = len(x) # D should have the same dimensions as each sequence length
            if D <= 1:  # TODO: data set has sequences length = 1, probably should filter these out
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
            #x[mask] = mask_id # not appending correctly :(
            mask = torch.tensor(mask, dtype=torch.bool)
            masks.append(mask)
            #mask = mask.to(torch.long)
            #print(mask.dtype, x.dtype, mask_id.dtype)
            x_t = ~mask[0:D] * x + mask[0:D] * mask_id
            #mask = torch.tensor(mask, dtype=torch.bool)
            src.append(x_t)
            #masks.append(mask)
        # PAD out
        src = _pad(src, self.tokenizer.pad_id)
        masks = _pad(masks*1,0) #, self.seq_length, 0)
        tokenized = _pad(tokenized, self.tokenizer.pad_id)
        #print("src shape",src.shape, "mask shape",masks.shape)
        return (src.to(torch.long), timesteps, tokenized.to(torch.long), masks)

class DMsMaskCollater(object):
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
    def __init__(self, simple_collater, tokenizer=Tokenizer(), masking_scheme="OA", inputs_padded=False, num_timesteps=100):
        self.simple_collater = simple_collater
        self.tokenizer = tokenizer
        self.inputs_padded  = inputs_padded
        self.masking_scheme = masking_scheme
        self.num_timesteps = num_timesteps # Only needed for markov trans, doesnt depend on seq len
        self.blosum = Blosum62()
        self.alphabet = [self.blosum.b_to_i[a] for a in BLOSUM62_ALPHABET]

    def __call__(self, sequences):
        tokenized = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in sequences]
        max_len = max(len(t) for t in tokenized)
        src=[]
        timesteps = []
        x_tgt = []
        masks=[]
        mask_id = torch.tensor(self.tokenizer.mask_id, dtype=torch.int64)

        if self.masking_scheme == "OA":
            for i, x in enumerate(tokenized):
                if self.inputs_padded:  # if truncating seqs to some length first in SimpleCollater, inputs will be padded
                    x_pad = x.clone()
                    mask_pad = x_pad != self.tokenizer.pad_id
                    # num_pad = len(x_pad) - mask_pad.sum()
                    x = x[mask_pad].to(torch.int64)
                # Randomly generate timestep and indices to mask
                D = len(x)  # D should have the same dimensions as each sequence length
                if D <= 1:  # TODO: data set has sequences length = 1, probably should filter these out
                    t = 1
                else:
                    t = np.random.randint(1, D)  # randomly sample timestep
                num_mask = (D - t + 1)  # from OA-ARMS
                # Append timestep
                timesteps.append(num_mask)
                # Generate mask
                mask_arr = np.random.choice(D, num_mask, replace=False)  # Generates array of len num_mask
                index_arr = np.arange(0, max_len)  # index array [1...seq_len]
                mask = np.isin(index_arr, mask_arr, invert=False).reshape(
                    index_arr.shape)  # mask bools indices specified by mask_arr
                # Mask inputs
                mask = torch.tensor(mask, dtype=torch.bool)
                masks.append(mask)
                x_t = ~mask[0:D] * x + mask[0:D] * mask_id
                src.append(x_t)
            # PAD out
            src = _pad(src, self.tokenizer.pad_id)
            masks = _pad(masks * 1, 0)  # , self.seq_length, 0)
            tokenized = _pad(tokenized, self.tokenizer.pad_id)
            return (src.to(torch.long), timesteps, tokenized.to(torch.long), masks)

        elif self.masking_scheme == "BLOSUM" or self.masking_scheme == "RANDOM":
            one_hot = [torch.LongTensor(self.blosum.one_hot(s[0])) for s in sequences]
            if self.masking_scheme == "BLOSUM":
                q = self.blosum.q_blosum
            elif self.masking_scheme == "RANDOM":
                q = self.blosum.q_random
            for i,x in enumerate(one_hot):
                if self.inputs_padded: # if truncating seqs to some length first in SimpleCollater, inputs will be padded
                     x_pad = x.clone()
                     mask_pad = x_pad != self.tokenizer.pad_id
                     x = x[mask_pad].to(torch.int64)
                # Randomly generate timestep and indices to mask
                D = len(x) # D should have the same dimensions as each sequence length
                t = np.random.randint(1, self.num_timesteps) # randomly sample timestep
                #print("t", t)
                #num_mask = (D-t+1) # from OA-ARMS
                # Append timestep
                timesteps.append(t)
                # Calculate target
                #print(x.shape, q.shape)
                x_next = sample_transition_matrix(x, q, t, self.alphabet)
                x_tgt.append(x_next)
                # Mask from input and output
                #print("here")
                #print(tokenized[i], x_next)
                mask_arr = _diff(tokenized[i],x_next)
                index_arr = np.arange(0, max_len)  # index array [1...seq_len]
                mask = np.isin(index_arr, mask_arr, invert=False).reshape(index_arr.shape)
                mask = torch.tensor(mask, dtype=torch.bool)
                masks.append(mask)
                # Create src
                x = ~mask[0:D] * tokenized[i] + mask[0:D] * mask_id
                src.append(x)
                #masks.append(mask)
            # PAD out
            src = _pad(src, self.tokenizer.pad_id)
            masks = _pad(masks*1, 0)
            #tokenized = _pad(tokenized, self.tokenizer.pad_id)
            x_tgt = _pad(x_tgt, self.tokenizer.pad_id)
            #print("src shape",src.shape, "mask shape",masks.shape)
            return (src.to(torch.long), timesteps, x_tgt.to(torch.long), masks)

        else:
            return None
            print("Choose OA, BLOSUM, or RANDOM as masking scheme")



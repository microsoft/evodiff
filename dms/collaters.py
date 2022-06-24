import numpy as np
import torch
from dms.utils import Tokenizer

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
        output[row, :len(t)] = t
    return output

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
    print(output.shape)
    for row, t in enumerate(tokenized):
        if len(t) <= seq_length:
            output[row, :len(t)] = t
        else:
            output[row, :seq_length] = t[:seq_length]
    return output

class SimpleCollater(object):
    """
    Slightly altered from protein-sequence-models (K. Yang)

    Performs simple operations on batch of sequences contained in a list
    - Can reverse sequence orders
    - Pad to "max" length in batch or pads to "normalized" length chosen by providing seq_length (default 512)
    """

    def __init__(self, pad=False, backwards=False, norm=False, seq_length=512):
        self.pad = pad
        self.seq_length = seq_length
        self.tokenizer = Tokenizer()
        self.backwards = backwards
        self.norm = norm

    def __call__(self, batch):
        prepped = self._prep(batch)
        return prepped

    def _prep(self, sequences):
        if self.backwards:
            sequences = [s[::-1] for s in sequences]

        tokenized = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in sequences]

        if self.norm:
            print("Truncating sequences to ", self.seq_length, "and padding")
            tokenized = _normalize_seq_lengths(tokenized, self.seq_length, self.tokenizer.pad_id)

        if self.pad:
            print("Padding sequences to max seq length")
            tokenized = _pad(tokenized, self.tokenizer.pad_id)

        print("Tokenizing sequences")
        return (tokenized)

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
        src=[]
        timesteps = []
        masks=[]
        mask_id = torch.tensor(self.tokenizer.mask_id, dtype=torch.int64)

        for i,x in enumerate(tokenized):
            if self.inputs_padded: # if truncating to some length first, inputs will be padded
                 x_pad = x.clone()
                 mask_pad = x_pad != self.tokenizer.pad_id
                 num_pad = len(x_pad) - mask_pad.sum()
                 x = x[mask_pad].to(torch.int64)
            # Randomly generate timestep and indices to mask
            D = len(x) # D should have the same dimensions as each sequence length
            t = np.random.randint(1, D) # randomly sample timestep
            num_mask = (D-t+1) # from OA-ARMS
            # Append timestep
            timesteps.append(num_mask)
            # Generate mask
            mask_arr = np.random.choice(D, num_mask, replace=False) # Generates array of len num_mask
            index_arr = np.arange(0, len(x)) #index array [1...seq_len]
            mask = np.isin(index_arr, mask_arr, invert=False).reshape(index_arr.shape) # mask bools indices specified by mask_arr
            # Mask inputs
            x[mask] = mask_id
            mask = torch.tensor(mask, dtype=torch.bool)
            src.append(x)
            masks.append(mask)
        # PAD out
        src = _pad(src, self.tokenizer.pad_id)
        masks = _pad(masks*1, 0)
        tokenized = _pad(tokenized, self.tokenizer.pad_id)
        return (src.to(torch.long), timesteps, tokenized.to(torch.long), masks)


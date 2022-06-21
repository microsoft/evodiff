## TODO: FIX/IMPLEMENT KEVIN NOTES

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

    for i, t in enumerate(tokenized):
        if len(t) <= seq_length:
            output[i, :len(t)] = t
        else:
            output[i, :seq_length] = t[:seq_length]
    return output

class SimpleCollater(object):
    """
    Slightly altered from protein-sequence-models (K. Yang)

    Performs simple operations on batch of sequences contained in a list
    - Can reverse sequence orders
    - Pad to "max" length in batch or pads to "normalized" length chosen by providing seq_length
    """

    def __init__(self, pad='Norm', backwards=False, seq_length=512):
        self.pad = pad
        self.seq_length = seq_length
        self.tokenizer = Tokenizer()
        self.backwards = backwards

    def __call__(self, batch):
        # for seq in batch:
        #     print("Length of sequence", len(seq))
        prepped = self._prep(batch)
        return prepped

    def _prep(self, sequences):
        if self.backwards:
            sequences = [s[::-1] for s in sequences]

        tokenized = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in sequences]

        if self.pad == 'Max':
            tokenized = _pad(tokenized, self.tokenizer.pad_id)
        elif self.pad == 'Norm':
            tokenized = _normalize_seq_lengths(tokenized, self.seq_length, self.tokenizer.pad_id)
        else:
            print("Must select 'Max' or 'Norm' for padding scheme")
        return (tokenized)

class OAMaskCollater(object):
    """
    OrderAgnosic Mask Collater for masking batch data according to Hoogeboom et al. OA ARDMS
    inputs:
        sequences : list of sequences
    OA-ARM variables:
        D : possible permutations from 0.. max length
        t : randomly selected timestep

    outputs:
        src : source (input) to model, list of masked sequenes
        timesteps: as a torch tensor (D-t+1) term
        tokenized: tokenized sequences
        masks: mask as int tensor
    """
    def __init__(self, simple_collater, mask_type='random', mask_id_letter='Y', mask_pad=False):
        self.simple_collater = simple_collater
        self.tokenizer = Tokenizer()
        self.mask_type = mask_type
        self.mask_id_letter= mask_id_letter
        self.mask_pad  = mask_pad

    def __call__(self, sequences):
        tokenized = self.simple_collater(sequences)
        src=[]
        timesteps = []
        masks=[]
        for i,x in enumerate(tokenized):
            num_pad = 0
            # TODO: padding order is inefficient, but buggy
            if not self.mask_pad: # if not masking pads ignore before creating mask_arr
                x_pad = x.clone()
                mask_pad = x_pad != self.tokenizer.pad_id
                x = x[mask_pad]
            D = len(x) # D should have the same dimensions as each sequence length
            t = np.random.randint(1,D) # randomly sample timestep
            num_mask = (D-t+1) # from OA-ARMS
            timesteps.append(num_mask)
            mask_arr = np.random.choice(D, num_mask, replace=False) # Generates array of len num_mask
            index_arr = np.arange(0, len(x))
            mask = np.isin(index_arr, mask_arr, invert=False).reshape(index_arr.shape) # masks indices specified by mask_arr
            mask = masks.append(mask)
            x[mask] = torch.Tensor(self.tokenizer.tokenize([self.mask_id_letter]))
            src.append(x)
        return (src, torch.tensor(timesteps), tokenized, torch.tensor(masks))


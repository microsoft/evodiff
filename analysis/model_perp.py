import numpy as np
from dms.pretrained import CARP_38M, CARP_640M, D3PM_BLOSUM_38M, D3PM_BLOSUM_640M, D3PM_UNIFORM_38M, D3PM_UNIFORM_640M,\
                           OA_AR_640M, OA_AR_38M, LR_AR_38M, LR_AR_640M, ESM1b_640M
from torch.nn import CrossEntropyLoss
from dms.losses import OAMaskedCrossEntropyLoss
from sequence_models.losses import MaskedCrossEntropyLoss
import torch
from sequence_models.datasets import UniRefDataset
from tqdm import tqdm
import pandas as pd
from analysis.plot import plot_perp_group_masked, plot_perp_group_d3pm
import math

def main():
    np.random.seed(0) # set random seed

    # Def read seqs from fasta
    data = UniRefDataset('data/uniref50/', 'rtest', structure=False, max_len=1022) # For ESM max_len=1022 (1024+start/stop), for DIFF 2048

    checkpoint = ESM1b_640M()
    save_name = 'esm1b_640M'

    losses = []
    n_tokens = []
    time_loss_data = []
    for i in tqdm(range(25000)): #len(data))):
        r_idx = np.random.choice(len(data))
        sequence = [data[r_idx]]
        t, loss, tokens = sum_nll_mask(sequence, checkpoint)
        #print(t, loss, tokens)
        if not np.isnan(loss): #esm-1b predicts nans at large % mask
            losses.append(loss)
            n_tokens.append(tokens)
            time_loss_data.append([t, loss, tokens])
        if i % 10000 == 0:
            ll = -sum(losses) / sum(n_tokens)
            perp = np.exp(-ll)
            print(i, "samples, perp:", np.mean(perp))
    print("Final test perp:", np.exp(sum(losses)/sum(n_tokens)))
    df = pd.DataFrame(time_loss_data, columns=['time', 'loss', 'tokens'])
    if checkpoint[-1] == 'd3pm':
        plot_perp_group_d3pm(df, save_name)
    else:
        plot_perp_group_masked(df, save_name)

def sum_nll_mask(sequence, checkpoint):
    model, collater, tokenizer, scheme = checkpoint
    # Use model.eval() if using CPU
    model.eval().cuda()

    # D3PM Collater returns; src, src_one_hot, timesteps, tokenized, tokenized_one_hot, Q, Q_bar, q_x
    if scheme == 'd3pm':
        src, src_onehot, timestep, tgt, tgt_onehot, Q, Q_bar, q = collater(sequence)
    elif scheme == 'mask' or scheme=='causal-mask':
        if scheme == 'mask':
            src, timestep, tgt, mask = collater(sequence)
        elif scheme == 'causal-mask':
            src, tgt, mask = collater(sequence)
        timestep = torch.tensor([0] * len(src))  # placeholder in model
        #print(sequence, tokenizer.untokenize(src[0]), tokenizer.untokenize(tgt[0]))
        input_mask = (src != tokenizer.pad_id).float() # Should be no pads since not batching
        mask = mask.cuda()
        input_mask = input_mask.cuda()
    elif scheme == 'esm-mask':
        src, timestep, tgt, mask = collater(sequence)
        input_mask = (src != tokenizer.padding_idx).float()  # Should be no pads since not batching
        mask = mask.cuda()
        input_mask = input_mask.cuda()
    src = src.cuda()     # Comment all variable.cuda() lines if using CPU
    timestep = timestep.cuda()
    tgt = tgt.cuda()
    with torch.no_grad():
        outputs = model(src, timestep) # for both d3pm and oaardm this is predicting x_0 (tgt)
        if scheme == 'esm-mask':
            outputs = outputs["logits"]

    # Get loss (NLL ~= CE)
    if scheme == 'd3pm':
        loss_func = CrossEntropyLoss(reduction='sum')
        nll_loss = loss_func(outputs.squeeze(), tgt.squeeze())
        #ll = -nll_loss.item() / len(tgt.squeeze()) # over all tokens
        t_out=timestep
        tokens = len(tgt.squeeze())
    elif scheme == 'mask' or scheme == 'esm-mask' or scheme=='causal-mask':
        if scheme=='causal-mask': # LR-AR only predict next token
            loss_func = MaskedCrossEntropyLoss(reduction='sum') # returns the mean
            tokens = mask.sum().item()
            nll_loss = loss_func(outputs, tgt, mask)
            #print(nll_loss, tokens)
            t_out=1
            #ll = -nll_loss.item()
        else:
            loss_func = OAMaskedCrossEntropyLoss(reweight=False)
            ce_loss, nll_loss = loss_func(outputs[:, :, :26], tgt, mask, timestep, input_mask) # returns a sum
            tokens = mask.sum().item()
            t_out = tokens / int(len(tgt.squeeze()))
            #ll = -nll_loss.item() / mask.sum().item() # over all masked tokens

    # Get perp
    #perp = np.exp(-ll)
    return t_out, nll_loss.item(), tokens

if __name__ == '__main__':
    main()
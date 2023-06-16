import numpy as np
from dms.pretrained import D3PM_BLOSUM_640M, OA_AR_640M
from torch.nn import CrossEntropyLoss
from dms.losses import OAMaskedCrossEntropyLoss
from dms.utils import parse_txt
import torch
from sequence_models.datasets import UniRefDataset
from tqdm import tqdm

def main():
    # Def read seqs from fasta
    data = UniRefDataset('data/uniref50/', 'rtest', structure=False)
    checkpoint = OA_AR_640M()

    perplexities = []
    for i in tqdm(range(len(data))):
        sequence = [data[i]]
        #print(sequence)
        p = calculate_perplexity(sequence, checkpoint)
        perplexities.append(p)
        if i % 10000 == 0:
            print(i, "samples, perp:", np.mean(perplexities))

def calculate_perplexity(sequence, checkpoint):
    model, collater, tokenizer, scheme = checkpoint
    # Use model.eval() if using CPU
    model.eval().cuda()

    # D3PM Collater returns; src, src_one_hot, timesteps, tokenized, tokenized_one_hot, Q, Q_bar, q_x
    if scheme == 'd3pm':
        src, src_onehot, timestep, tgt, tgt_onehot, Q, Q_bar, q = collater(sequence)
    elif scheme == 'mask':
        src, timestep, tgt, mask = collater(sequence)
        input_mask = (src != tokenizer.pad_id).float() # Should be no pads since not batching
        # Comment all variable.cuda() lines if using CPU
        mask = mask.cuda()
        input_mask = input_mask.cuda()
    # Comment all variable.cuda() lines if using CPU
    src = src.cuda()
    timestep = timestep.cuda()
    tgt = tgt.cuda()
    with torch.no_grad():
        outputs = model(src, timestep) # for both d3pm and oaardm this is predicting x_0 (tgt)

    # Get loss (NLL ~= CE)
    if scheme == 'd3pm':
        loss_func = CrossEntropyLoss(reduction='sum')
        nll_loss = loss_func(outputs.squeeze(), tgt.squeeze())
        ll = -nll_loss.item() / len(tgt.squeeze()) # over all tokens
    elif scheme == 'mask':
        loss_func = OAMaskedCrossEntropyLoss(reweight=False)
        ce_loss, nll_loss = loss_func(outputs, tgt, mask, timestep, input_mask)  # returns a sum of losses
        ll = -nll_loss.item() / mask.sum().item() # over masked tokens
    # Get perp
    perp = np.exp(-ll)
    #print("num mask:", int(mask.sum().item()), "of total:", int(len(tgt.squeeze())), f', Perplexity: {perp:.5f}')
    #print("t:",timestep, f', Perplexity: {perp:.5f}')
    return perp

if __name__ == '__main__':
    main()
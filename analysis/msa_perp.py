import numpy as np
import evodiff.utils
from evodiff.pretrained import MSA_D3PM_UNIFORM_RANDSUB, MSA_D3PM_UNIFORM_MAXSUB, MSA_D3PM_BLOSUM_RANDSUB, \
    MSA_D3PM_BLOSUM_MAXSUB, MSA_OA_DM_RANDSUB, MSA_OA_DM_MAXSUB, ESM_MSA_1b
from evodiff.losses import D3PMCELoss
from sequence_models.losses import MaskedCrossEntropyLossMSA
import torch
from tqdm import tqdm
import pandas as pd
from evodiff.plot import plot_perp_group_masked, plot_perp_group_d3pm
import argparse
import os

def main():
    # set seeds
    _ = torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='D3PM_BLOSUM_38M',
                        help='Choice of: msa_d3pm_uniform_randsub, msa_d3pm_uniform_maxsub,\
                         msa_d3pm_blosum_randsub, msa_d3pm_blosum_maxsub,\
                         msa_oa_dm_randsub, msa_oa_dm_maxsub, esm_msa_1b')
    parser.add_argument('--subsampling', type=str, default='random') # or MaxHamming
    args = parser.parse_args()

    save_name = args.model_type

    if args.model_type=='msa_d3pm_uniform_randsub':
        checkpoint = MSA_D3PM_UNIFORM_RANDSUB()
    elif args.model_type=='msa_d3pm_uniform_maxsub':
        checkpoint = MSA_D3PM_UNIFORM_MAXSUB()
    elif args.model_type=='msa_d3pm_blosum_randsub':
        checkpoint = MSA_D3PM_BLOSUM_RANDSUB()
    elif args.model_type=='msa_d3pm_blosum_maxsub':
        checkpoint = MSA_D3PM_BLOSUM_MAXSUB()
    elif args.model_type=='msa_oa_dm_randsub':
        checkpoint = MSA_OA_DM_RANDSUB()
    elif args.model_type=='msa_oa_dm_maxsub':
        checkpoint = MSA_OA_DM_MAXSUB()
    elif args.model_type=='esm_msa_1b':
        checkpoint = ESM_MSA_1b()
    else:
        print("Please select valid model, i don't understand:", args.model_type)
    try:
        data_top_dir = os.getenv('AMLT_DATA_DIR') + '/data/data/data/'
    except:
        data_top_dir = 'data/'

    num_seqs=2000

    data = evodiff.data.get_valid_msas(data_top_dir, data_dir='openfold/', selection_type=args.subsampling, n_sequences=64, max_seq_len=512,
                                       out_path='../evodiff/ref/')

    losses = []
    n_tokens = []
    time_loss_data = []
    for i in tqdm(range(num_seqs)): #len(data))):
        r_idx = np.random.choice(len(data))
        sequence = [data[r_idx]]
        t, loss, tokens = sum_nll_mask(sequence, checkpoint)
        if not np.isnan(loss): #esm-1b predicts nans at large % mask
            losses.append(loss)
            n_tokens.append(tokens)
            if args.model_type == 'msa_oa_ar_randsub' or args.model_type == 'msa_oa_ar_maxsub' or args.model_type =='esm_msa_1b':
                time_loss_data.append([t, loss, tokens])
            else:
                time_loss_data.append([t.item(), loss, tokens])
        if i % 1000 == 0:
            ll = -sum(losses) / sum(n_tokens)
            perp = np.exp(-ll)
            print(i, "samples, perp:", np.mean(perp))
    print("Final test perp:", np.exp(sum(losses)/sum(n_tokens)))
    df = pd.DataFrame(time_loss_data, columns=['time', 'loss', 'tokens'])
    df.to_csv('plots/perp_df_' + save_name + '.csv')
    if checkpoint[-1] == 'd3pm':
        plot_perp_group_d3pm(df, save_name)
    else:
        plot_perp_group_masked(df, save_name)

def sum_nll_mask(sequence, checkpoint):
    model, collater, tokenizer, scheme = checkpoint
    model.eval().cuda() # Use model.eval() if using CPU

    # D3PM Collater returns; src, src_one_hot, timesteps, tokenized, tokenized_one_hot, Q, Q_bar, q_x
    if scheme == 'd3pm':
        src, src_onehot, timestep, tgt, tgt_onehot, Q, Q_bar, q = collater(sequence)
        input_mask = (src != tokenizer.pad_id).float() # placeholder
        input_mask = input_mask.cuda()
        timestep = timestep.cuda()
    elif scheme == 'mask':
        src, tgt, mask = collater(sequence)
        input_mask = (src != tokenizer.pad_id).float() # placeholder, should be no pads since not batching
        mask = mask.cuda()
        input_mask = input_mask.cuda()
    elif scheme == 'esm-mask':
        src, tgt, mask = collater(sequence)
        input_mask = (src != tokenizer.padding_idx).float()  # placeholder, should be no pads since not batching
        mask = mask.cuda()
        input_mask = input_mask.cuda()
    src = src.cuda()     # Comment all variable.cuda() lines if using CPU
    tgt = tgt.cuda()
    with torch.no_grad():
        if scheme == 'd3pm':
            outputs = model(src, timestep) # outputs are x_tilde_0 (predicted tgt)
        elif scheme == 'esm-mask':
            outputs = model(src, repr_layers=[33], return_contacts=True)
            outputs = outputs["logits"]
        else:
            outputs = model(src)

    # Get loss (NLL ~= CE)
    if scheme == 'd3pm':
        loss_func = D3PMCELoss(reduction='sum',tokenizer=tokenizer, sequences=False)
        nll_loss = loss_func(outputs, tgt, input_mask)
        t_out=timestep
        tokens_msa = tgt.squeeze().shape
        tokens = tokens_msa[0]*tokens_msa[1]
    elif scheme == 'mask' or scheme == 'esm-mask':
        if scheme == 'esm-mask':
            loss_func = MaskedCrossEntropyLossMSA(ignore_index=tokenizer.padding_idx, reweight=False)
        else:
            loss_func = MaskedCrossEntropyLossMSA(ignore_index=tokenizer.pad_id, reweight=False)
        ce_loss, nll_loss = loss_func(outputs, tgt, mask, input_mask) # returns a sum
        tokens = mask.sum().item()
        t_out = tokens / int(tgt.squeeze().shape[0]*tgt.squeeze().shape[1])
    return t_out, nll_loss.item(), tokens # return timestep sampled (or % masked), sum of losses, and sum of tokens

if __name__ == '__main__':
    main()
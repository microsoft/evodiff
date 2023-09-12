import numpy as np
import argparse
import torch
import os
import glob
import random
from evodiff.utils import Tokenizer
import pathlib
from sequence_models.datasets import UniRefDataset
from tqdm import tqdm
from evodiff.plot import aa_reconstruction_parity_plot
import pandas as pd
from evodiff.pretrained import CARP_38M, CARP_640M, D3PM_BLOSUM_38M, D3PM_BLOSUM_640M, D3PM_UNIFORM_38M, D3PM_UNIFORM_640M,\
                           OA_DM_640M, OA_DM_38M, LR_AR_38M, LR_AR_640M, ESM1b_650M


home = str(pathlib.Path.home())

def main():
    # set seeds
    _ = torch.manual_seed(0)
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='oa_dm_640M',
                        help='Choice of: carp_38M carp_640M esm1b_650M \
                                 oa_dm_38M oa_dm_640M lr_ar_38M lr_ar_640M d3pm_blosum_38M d3pm_blosum_640M d3pm_uniform_38M d3pm_uniform_640M')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    #parser.add_argument('out_fpath', type=str, nargs='?', default=os.getenv('PT_OUTPUT_DIR', '/tmp') + '/')
    parser.add_argument('--num-seqs', type=int, default=20)
    parser.add_argument('--penalty', type=float, default=None) # repetition penalty, commonly 1.2 is used
    parser.add_argument('--delete-prev',action='store_true')  # Will delete previous generated sequences
    parser.add_argument('--count', default=0, type=int) # Start new gen sequences from 0, this is when appending new seqs to files
    parser.add_argument('--scheme', default=None, type=str,
                        help='use train-sample valid-sample test-sample or random to generate samples not from model')
    parser.add_argument('--amlt', action='store_true')
    parser.add_argument('--random-baseline', action='store_true')
    args = parser.parse_args()

    data = UniRefDataset('data/uniref50/', 'train', structure=False, max_len=2048)
    data_valid = UniRefDataset('data/uniref50/', 'rtest', structure=False, max_len=2048)

    d3pm = False
    if args.model_type=='esm1b_650M':
        checkpoint = ESM1b_650M()
    elif args.model_type=='carp_38M':
        checkpoint = CARP_38M()
    elif args.model_type=='carp_640M':
        checkpoint = CARP_640M()
    elif args.model_type=='oa_dm_38M':
        checkpoint = OA_DM_38M()
    elif args.model_type=='oa_dm_640M':
        checkpoint = OA_DM_640M()
    elif args.model_type=='lr_ar_38M':
        checkpoint = LR_AR_38M()
    elif args.model_type=='lr_ar_640M':
        checkpoint = LR_AR_640M()
    elif args.model_type=='d3pm_blosum_38M':
        checkpoint = D3PM_BLOSUM_38M(return_all=True)
        d3pm=True
    elif args.model_type=='d3pm_blosum_640M':
        checkpoint = D3PM_BLOSUM_640M(return_all=True)
        d3pm=True
    elif args.model_type == 'd3pm_uniform_38M':
        checkpoint = D3PM_UNIFORM_38M(return_all=True)
        d3pm=True
    elif args.model_type == 'd3pm_uniform_640M':
        checkpoint = D3PM_UNIFORM_640M(return_all=True)
        d3pm=True
    else:
        raise Exception("Please select either carp_38M, carp_640M, esm1b_650M, oa_dm_38M, oa_dm_640M, lr_ar_38M, lr_ar_640M, d3pm_blosum_38M, d3pm_blosum_640M, d3pm_uniform_38M, or d3pm_uniform_640M. You selected:", args.model_type)

    if d3pm:
        model, collater, tokenizer, scheme, timestep, Q_bar, Q = checkpoint
    else:
        model, collater, tokenizer, scheme = checkpoint

    torch.cuda.set_device(args.gpus)
    device = torch.device('cuda:' + str(args.gpus))
    model = model.eval().to(device)

    # Out directories
    if args.amlt:
        home = os.getenv('AMLT_OUTPUT_DIR', '/tmp') + '/'
        top_dir = ''
        out_fpath = home
    else:
        home = str(pathlib.Path.home()) + '/Desktop/DMs/'
        top_dir = home
        if not args.random_baseline:
            out_fpath = home + args.model_type + '/'
        else:
            scheme='random'
            out_fpath = home + 'random-baseline/'

    if not os.path.exists(out_fpath):
        os.makedirs(out_fpath)

    data_top_dir = top_dir + 'data/'

    # Delete prev runs
    if args.delete_prev:
        filelist = glob.glob(out_fpath+'generated*')
        for file in filelist:
            os.remove(file)
            print("Deleting", file, "in", out_fpath)

    # Run generation
    if scheme == 'causal-mask':
        sample, string = generate_autoreg(model, tokenizer, samples=args.num_seqs, penalty=args.penalty, device=device)

    elif scheme == 'test-sample':
        string = generate_valid_subset(data_valid, samples=args.num_seqs)

    elif scheme == 'random':
        train_prob_dist = aa_reconstruction_parity_plot(home, out_fpath, 'placeholder.csv', gen_file=False)
        string = []
        for _ in tqdm(range(args.num_seqs)):
            r_idx = np.random.choice(len(data))
            seq_len = len(data[r_idx][0])  # randomly sample a sequence length from train data
            i_string = generate_random_seq(seq_len, train_prob_dist)
            print(i_string)
            string.append([i_string])
    else:
        string = []
        sample = []
        for _ in tqdm(range(args.num_seqs)):
            r_idx = np.random.choice(len(data))
            seq_len = len(data[r_idx][0])  # randomly sample a sequence length from train data

            if scheme == 'mask':
                i_sample, i_string = generate_oaardm(model, tokenizer, seq_len, penalty=args.penalty, batch_size=1, device=device)
            elif scheme == 'd3pm':
                i_sample, i_string = generate_d3pm(model, tokenizer, Q, Q_bar, timestep, seq_len, batch_size=1,
                                                   device=device)
            string.append(i_string)
            sample.append(i_sample)
    print("String", string)
    # Write list of sequences (string) to fasta and CSV
    with open(out_fpath + 'generated_samples_string.csv', 'w') as f:
        for _s in string:
            f.write(_s[0] + "\n")
    with open(out_fpath + 'generated_samples_string.fasta', 'w') as f:
        for i, _s in enumerate(string):
            f.write(">SEQUENCE_" + str(args.count+i) + "\n" + str(_s[0]) + "\n")

    # Plot distribution of generated samples
    aa_reconstruction_parity_plot(home, out_fpath, 'generated_samples_string.csv')


def generate_oaardm_order_opt(model, tokenizer, seq_len, penalty=None, batch_size=20, device='cuda'):
    # Generate a random start string and convert to tokens
    all_aas = tokenizer.all_aas
    mask = tokenizer.mask_id
    # Start from mask
    sample = torch.zeros((batch_size, seq_len))+mask
    sample = sample.to(torch.long)
    sample = sample.to(device)
    loc = np.arange(seq_len)
    timestep = torch.tensor([0] * batch_size)  # placeholder but not used in model
    timestep = timestep.to(device)
    with torch.no_grad():
        for _ in loc:
            # Prob-based loc sampling
            prediction = model(sample, timestep) # output shape B x L x T
            p = torch.nn.functional.softmax(prediction[:, :, :len(all_aas) - 6], dim=-1) # normalize along L dim
            nonmask_loc = (sample != mask).unsqueeze(-1).expand(p.shape[0], p.shape[1], p.shape[2])
            p[nonmask_loc] = 0 # ignore tokens that have already been sampled
            idx = torch.argmax(p.view(1, -1), dim=-1)
            pos = torch.div(idx, p.shape[-1],rounding_mode='trunc')
            #aas = idx % p.shape[-1] # for argmax use this
            aas = torch.multinomial(p[0, pos], num_samples=1) # argmax looks bad, sample at each confident pos
            #print(idx, pos, aas)
            sample[:, pos] = aas
            #print("pos", pos.item(), [tokenizer.untokenize(s) for s in sample])
    untokenized = [tokenizer.untokenize(s) for s in sample]
    return sample, untokenized


def generate_oaardm(model, tokenizer, seq_len, penalty=None, batch_size=3, device='cuda'):
    # Generate a random start string and convert to tokens
    all_aas = tokenizer.all_aas
    mask = tokenizer.mask_id

    # Start from mask
    sample = torch.zeros((batch_size, seq_len))+mask
    sample = sample.to(torch.long)
    sample = sample.to(device)

    # Unmask 1 loc at a time randomly
    loc = np.arange(seq_len)
    np.random.shuffle(loc)
    with torch.no_grad():
        for i in tqdm(loc):
            timestep = torch.tensor([0] * batch_size) # placeholder but not called in model
            timestep = timestep.to(device)
            prediction = model(sample, timestep) #, input_mask=input_mask.unsqueeze(-1)) #sample prediction given input
            p = prediction[:, i, :len(all_aas)-6] # sample at location i (random), dont let it predict non-standard AA
            p = torch.nn.functional.softmax(p, dim=1) # softmax over categorical probs
            p_sample = torch.multinomial(p, num_samples=1)
            # Repetition penalty
            if penalty is not None: # ignore if value is None
                for j in range(batch_size): # iterate over each obj in batch
                    case1 = (i == 0 and sample[j, i+1] == p_sample[j]) # beginning of seq
                    case2 = (i == seq_len-1 and sample[j, i-1] == p_sample[j]) # end of seq
                    case3 = ((i < seq_len-1 and i > 0) and ((sample[j, i-1] == p_sample[j]) or (sample[j, i+1] == p_sample[j]))) # middle of seq
                    if case1 or case2 or case3:
                        #print("identified repeat", p_sample, sample[i-1], sample[i+1])
                        p[j, int(p_sample[j])] /= penalty # reduce prob of that token by penalty value
                        p_sample[j] = torch.multinomial(p[j], num_samples=1) # resample
            sample[:, i] = p_sample.squeeze()
            #print([tokenizer.untokenize(s) for s in sample]) # check that sampling correctly
    #print("final seq", [tokenizer.untokenize(s) for s in sample])
    untokenized = [tokenizer.untokenize(s) for s in sample]
    return sample, untokenized

def generate_autoreg(model, tokenizer, samples=100, batch_size=1, max_seq_len=1024):
    # Generates 1 seq at a time, no batching, to make it easier to deal w variable seq lengths
    # Generates until max length or until stop token is predicted
    #model.eval().cuda()
    device = model.device()

    start = tokenizer.start_id
    stop = tokenizer.stop_id
    sample_out = []
    untokenized_out = []
    timestep = torch.tensor([0] * batch_size)  # placeholder but not called in model
    timestep = timestep.to(device)
    for s in tqdm(range(samples)):
        # Start from START token
        sample = (torch.zeros((1))+ start).unsqueeze(0) # add batch dim
        sample = sample.to(torch.long)
        sample = sample.to(device)
        # Iterate over each residue until desired length
        #max_loc = np.arange(max_seq_len)
        reach_stop=False # initialize
        with torch.no_grad():
            for i in range(max_seq_len):
                if reach_stop == False: # Add residues until it predicts STOP token or hits max seq len
                    prediction = model(sample, timestep) #, input_mask=input_mask.unsqueeze(-1)) #sample prediction given input
                    p = prediction[:, -1, :] # predict next token
                    p = torch.nn.functional.softmax(p, dim=1) # softmax over categorical probs
                    p_sample = torch.multinomial(p, num_samples=1)
                    sample = torch.cat((sample, p_sample), dim=1)
                    #print(tokenizer.untokenize(sample[0]))
                    #print(p_sample, stop)
                    if p_sample == stop:
                        reach_stop = True
                else:
                    break

        print("final seq", tokenizer.untokenize(sample[0,1:-1])) # dont save start/stop tokens
        untokenized = tokenizer.untokenize(sample[0,1:-1])
        sample_out.append(sample[0,1:-1])
        untokenized_out.append(untokenized)
    return sample_out, untokenized_out


def generate_d3pm(model, tokenizer, Q, Q_bar, timesteps, seq_len, batch_size=3, device='cuda'):
    """
    Generate a random start string from uniform dist and convert to predictions
    """
    #model.eval()
    #device = model.device()

    sample = torch.randint(0, tokenizer.K, (batch_size, seq_len))
    sample = sample.to(torch.long)
    sample = sample.to(device)
    Q = Q.to(device)
    Q_bar = Q_bar.to(device)

    timesteps = torch.linspace(timesteps-1,1,int((timesteps-1)/1), dtype=int) # iterate over reverse timesteps
    timesteps = timesteps.to(device)
    with torch.no_grad():
        for t in tqdm(timesteps):
            timesteps = torch.tensor([t] * batch_size)
            timesteps = timesteps.to(device)
            prediction = model(sample, timesteps)
            p = prediction[:, :, :tokenizer.K]  # p_theta_tilde (x_0_tilde | x_t) # Don't predict non-standard AAs
            p = torch.nn.functional.softmax(p, dim=-1)  # softmax over categorical probs
            p = p.to(torch.float64)
            x_tminus1 = sample.clone()
            for i, s in enumerate(sample):
                x_t_b = tokenizer.one_hot(s)
                A = torch.mm(x_t_b, torch.t(Q[t]))  # [P x K]
                Q_expand = Q_bar[t-1].unsqueeze(0).expand(A.shape[0], tokenizer.K, tokenizer.K)  # [ P x K x K]
                B_pred = torch.mul(p[i].unsqueeze(2), Q_expand)
                q_t = torch.mul(A.unsqueeze(1), B_pred)  # [ P x K x K ]
                p_theta_marg = torch.bmm(torch.transpose(q_t, 1,2),  p[i].unsqueeze(2)).squeeze()  # this marginalizes over dim=2
                p_theta_marg = p_theta_marg / p_theta_marg.sum(axis=1, keepdim=True)
                x_tminus1[i] = torch.multinomial(p_theta_marg, num_samples=1).squeeze()
                # On final timestep pick next best from standard AA
                if t == 1:
                     x_tminus1[i] = torch.multinomial(p_theta_marg[:, :tokenizer.K-6], num_samples=1).squeeze()
                # diff = torch.ne(s, x_tminus1[i])
                # if t % 100 == 0:
                #     print("time", t, diff.sum().item(), "mutations", tokenizer.untokenize(x_tminus1[i]), "sample", tokenizer.untokenize(s))
            sample = x_tminus1

    untokenized = [tokenizer.untokenize(s) for s in sample]
    print("final seq", untokenized)
    return sample, untokenized

def generate_random_seq(seq_len, train_prob_dist, tokenizer=Tokenizer()):
    """
    Generates a set of random sequences drawn from a train distribution
    """
    sample = torch.multinomial(torch.tensor(train_prob_dist), num_samples=seq_len, replacement=True)
    sample = sample.to(torch.long)
    return tokenizer.untokenize(sample)

def generate_valid_subset(data_valid, samples=20):
    sample = []
    for i in tqdm(range(samples)):
        r_idx = np.random.choice(len(data_valid))
        sequence = data_valid[r_idx][0]
        sample.append(sequence)
    print(sample)
    return sample


if __name__ == '__main__':
    main()
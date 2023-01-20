import argparse
import json
import os
import numpy as np
import torch
from sequence_models.esm import MSATransformer
from sequence_models.constants import MSA_ALPHABET, MSA_PAD, MASK
from dms.utils import Tokenizer
from dms.model import MSATransformerTime
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_fpath')
    parser.add_argument('out_fpath', type=str, nargs='?',
                        default='/Users/nityathakkar/Desktop/research/msr/model_output/')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-off', '--offset', default=0, type=int,
                        help='Number of GPU devices to skip.')
    parser.add_argument('-sd', '--state_dict', default=None)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--final_norm', action='store_true')
    parser.add_argument('--mask', type=str, default='autoreg')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--n-sequences', type=int, default=64)
    parser.add_argument('--seq-length', type=int, default=256)
    parser.add_argument('--gen_task', type=str, default='masked')
    args = parser.parse_args()

    _ = torch.manual_seed(0)
    np.random.seed(0)

    torch.cuda.set_device(args.gpus + args.offset)
    device = torch.device('cuda:' + str(args.gpus + args.offset))
    #device = torch.device("cpu")
    with open(args.config_fpath, 'r') as f:
        config = json.load(f)

    d_embed = config['d_embed']
    d_hidden = config['d_hidden']
    n_layers = config['n_layers']
    n_heads = config['n_heads']

    try:
        data_top_dir = os.getenv('AMLT_DATA_DIR') + '/'
        data_dir = os.getenv('AMLT_DATA_DIR') + '/'
        data_dir += config['dataset'] + '/'
        ptjob = True
    except:
        data_top_dir = 'data/'
        #print(data_top_dir)
        data_dir = data_top_dir
        data_dir += config['dataset'] + '/'
        ptjob = False

    if args.mask == 'autoreg':
        tokenizer = Tokenizer()
        diffusion_timesteps = None # Not input to model
    elif args.mask == 'blosum' or args.mask == 'random':
        diffusion_timesteps = config['diffusion_timesteps']
        tokenizer = Tokenizer(path_to_blosum=data_top_dir+"blosum62-special-MSA.mat")
        if args.mask == 'random':
            Q_prod, Q_t = tokenizer.q_random_schedule(timesteps=diffusion_timesteps)
        if args.mask == 'blosum':
            Q_prod, Q_t = tokenizer.q_blosum_schedule(timesteps=diffusion_timesteps)
        Q_prod = Q_prod.to(device)
        Q_t = Q_t.to(device)
    else:
        print("mask must be: 'autoreg', 'blosum', or 'random'")

    padding_idx = tokenizer.pad_id  # PROTEIN_ALPHABET.index(PAD)
    masking_idx = tokenizer.mask_id
    if args.mask == 'autoreg':
        tokenizer = Tokenizer()
        model = MSATransformer(d_embed, d_hidden, n_layers, n_heads, use_ckpt=True, n_tokens=len(MSA_ALPHABET),
                               padding_idx=MSA_ALPHABET.index(MSA_PAD), mask_idx=MSA_ALPHABET.index(MASK)).cuda()
    else:
        model = MSATransformerTime(d_embed, d_hidden, n_layers, n_heads, timesteps=diffusion_timesteps, use_ckpt=True,
                                   n_tokens=len(MSA_ALPHABET), padding_idx=padding_idx, mask_idx=masking_idx).cuda()

    model = model.to(device)

    # Restore the model weights for the last checkpoint after training
    outputs = os.listdir(args.out_fpath)
    if len(outputs) > 0:
       last_epoch = 0
       for output in outputs:
           if 'checkpoint' in output:
               epoch = int(output.split('checkpoint')[-1][:-4])
               if epoch > last_epoch:
                   args.state_dict = args.out_fpath + output
                   last_epoch = epoch

    print('Using checkpoint', last_epoch)
    print('Loading weights from ' + args.state_dict + '...')
    sd = torch.load(args.state_dict, map_location=torch.device('cpu'))
    msd = sd['model_state_dict']
    msd = {k.split('module.')[1]: v for k, v in msd.items()}
    model.load_state_dict(msd)

    seqs = ""
    with open(args.out_fpath + 'generated_samples_string.fasta', 'a') as f:
        count = 0
        fasta_string = ""
        if args.mask == 'autoreg':
            sample, string = generate_msa(model, tokenizer, args.batch_size, args.n_sequences, args.seq_length,
                                          device=device)
        if args.mask == 'blosum' or args.mask=='random':
            sample, string = generate_msa_d3pm(model, args.batch_size, args.n_sequences, args.seq_length,
                                                 Q_bar=Q_prod, Q=Q_t, tokenizer=Tokenizer(), timesteps=diffusion_timesteps,
                                                 no_step=False, device=device)
        for _s in string:
            fasta_string += ">SEQUENCE_" + str(count) + "\n" + str(_s[0]) + "\n"
            count += 1
            seqs += str(_s[0]) + "\n"

        f.write(fasta_string)
        f.close()

    with open(args.out_fpath + 'generated_samples_string.csv', 'a') as f:
        f.write(','.join(
            [seqs]))
        f.write('\n')

def save_msa_a3m(msa_string, file): # TODO debug = make sure this works, i think you are missing a nested list soemwhere
    with open(file, 'a') as f:
        for msa in msa_string:
            f.write('>query \n')
            msa[0]
            for seq in msa:
                f.write('\n')
                f.write('>tr \n')
                f.write(seq)

def generate_msa(model, tokenizer, batch_size, n_sequences, seq_length, device='gpu'):
    mask_id = tokenizer.mask_id
    src = torch.full((batch_size, n_sequences, seq_length), fill_value=mask_id)
    src = src.to(device)
    output = src.clone()
    #print("input seq", tokenizer.untokenize(output[0].flatten()))

    masked_loc_x = np.arange(n_sequences)
    masked_loc_y = np.arange(seq_length)

    all_ind = np.transpose([np.tile(masked_loc_x, len(masked_loc_y)), np.repeat(masked_loc_y, len(masked_loc_x))])
    np.random.shuffle(all_ind)

    with torch.no_grad():
        for i in tqdm(all_ind):
            random_x, random_y = i
            preds = model(output)  # Output shape of preds is (BS=1, N=64, L, n_tokens=31)
            p = preds[:, random_x, random_y, :]
            if random_x == 0 : # for first row don't let p_softmax predict gaps
                p = preds[:, random_x, random_y, :tokenizer.K-1]
            p_softmax = torch.nn.functional.softmax(p, dim=1)
            p_sample = torch.multinomial(input=p_softmax, num_samples=1)
            p_sample = p_sample.squeeze()
            output[:, random_x, random_y] = p_sample
            #print("time", random_x, random_y, "sample", tokenizer.untokenize(output[0].flatten()))
    output_ret1 = output
    output_ret2 = [[tokenizer.untokenize(s) for s in msa] for msa in output_ret1]
    print("final seq", output_ret2[0][0][:seq_length])
    return output_ret1, output_ret2 # return output and untokenized output

def generate_msa_d3pm(model, batch_size, n_sequences, seq_length, Q_bar=None, Q=None, tokenizer=Tokenizer(),
                      timesteps=500, no_step=False, device='gpu'):

    sample = torch.randint(0, tokenizer.K, (batch_size, n_sequences, seq_length)) # don't include gap token?
    sample = sample.to(torch.long)
    sample = sample.to(device)
    print("input query seq", tokenizer.untokenize(sample[0].flatten()[:seq_length]))
    print(sample.shape)
    if no_step:
        timesteps = np.linspace(timesteps-1, timesteps-1, 1, dtype=int)
    else:
        timesteps = np.linspace(timesteps-1,1,int((timesteps-1)/1), dtype=int) # iterate over reverse timesteps
    with torch.no_grad():
        for t in tqdm(timesteps):
            timesteps = torch.tensor([t] * batch_size)
            timesteps = timesteps.to(device)
            prediction = model(sample, timesteps)
            p = prediction[:, :, :, :tokenizer.K]  # p_theta_tilde (x_0_tilde | x_t)
            p = torch.nn.functional.softmax(p, dim=-1)  # softmax over categorical probs
            p = p.to(torch.float64)
            if no_step: # This one-step model ignores step-wise generation, will do better as lambda is larger
                x_tminus1 = sample.clone()
                for i in range(len(p)):
                    p_current = p[i].flatten(start_dim=0, end_dim=1)
                    x_tminus1[i] = torch.multinomial(p_current, num_samples=1).squeeze().reshape(n_sequences, seq_length)
            else:
                x_tminus1 = sample.clone()
                for i, s in enumerate(sample): # iterate over batches
                    # Calculate p_theta_marg from p_theta_tilde
                    x_t_b = torch.stack([tokenizer.one_hot(s_i) for s_i in s])
                    x_t_b = x_t_b.flatten(start_dim=0, end_dim=1)
                    p_current = p[i].flatten(start_dim=0, end_dim=1)
                    A = torch.mm(x_t_b, torch.t(Q[t]))  # [P x K]
                    Q_expand = Q_bar[t-1].unsqueeze(0).expand(A.shape[0], tokenizer.K, tokenizer.K)  # [ P x K x K]
                    B_pred = torch.mul(p_current.unsqueeze(2), Q_expand)
                    q_t = torch.mul(A.unsqueeze(1), B_pred)  # [ P x K x K ]
                    p_theta_marg = torch.bmm(torch.transpose(q_t, 1,2),  p_current.unsqueeze(2)).squeeze()  # this marginalizes over dim=2
                    p_theta_marg = p_theta_marg / p_theta_marg.sum(axis=1, keepdim=True)
                    x_tminus1_temp = torch.multinomial(p_theta_marg[:, :], num_samples=1).squeeze()
                    x_tminus1[i] = x_tminus1_temp.reshape(n_sequences, seq_length)
                    #diff = torch.ne(s, x_tminus1[i])
                    #if t % 100 == 0:
                    #    print("time", t, diff.sum().item(), "mutations") #, tokenizer.untokenize(x_tminus1))
                    #    print("query", tokenizer.untokenize(x_tminus1_temp[:seq_length]))
                sample = x_tminus1
    untokenized = [[tokenizer.untokenize(sample[i].flatten())] for i in range(batch_size)]
    print(len(untokenized[0]))
    print(len(untokenized[0][0]))
    print("final seq", untokenized[0][0][:seq_length])
    return sample, untokenized



if __name__ == '__main__':
    main()
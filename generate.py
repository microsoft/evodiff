from dms.model import ByteNetLMTime
import numpy as np
import argparse
from sequence_models.constants import MSA_ALPHABET, ALL_AAS, PROTEIN_ALPHABET, PAD
import torch
import os
import json
from dms.utils import Tokenizer
import pathlib

### SET RANDOM SEEDS ####
random_seed = 1
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)
torch.cuda.empty_cache()  # empty caches

home = str(pathlib.Path.home())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_fpath')
    parser.add_argument('out_fpath', type=str, nargs='?', default=os.getenv('PT_OUTPUT_DIR', '/tmp') + '/')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--final_norm', action='store_true')
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--num_seqs', type=int, default=100)
    parser.add_argument('--mask', type=str, default='autoreg')
    parser.add_argument('--penalty', type=float, default=None) # repetition penalty, commonly 1.2 is used
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--no_step', action='store_true') # For D3PM if true will predict x_0 from x_t, instead of x_tminus1
    args = parser.parse_args()

    with open(args.config_fpath, 'r') as f:
        config = json.load(f)

    d_embed = config['d_embed']
    d_model = config['d_model']
    n_layers = config['n_layers']
    kernel_size = config['kernel_size']
    r = config['r']
    if 'rank' in config:
        weight_rank = config['rank']
    else:
        weight_rank = None
    if 'slim' in config:
        slim = config['slim']
    else:
        slim = True
    if 'activation' in config:
        activation = config['activation']
    else:
        activation = 'relu'
    data_top_dir = home + '/Desktop/DMs/data/'

    torch.cuda.set_device(args.gpus)

    causal = False
    n_tokens = len(MSA_ALPHABET)

    if args.mask == 'autoreg' or args.mask == 'so':
        tokenizer = Tokenizer()
        diffusion_timesteps = None  # Not input to model
        if args.mask == 'so':
            n_tokens = len(PROTEIN_ALPHABET)
            tokenizer = Tokenizer(protein_alphabet=PROTEIN_ALPHABET, all_aas=ALL_AAS, pad=PAD)
            causal = True
    elif args.mask == 'blosum' or args.mask == 'random':
        tokenizer = Tokenizer(path_to_blosum=data_top_dir + "blosum62-special-MSA.mat")
        diffusion_timesteps = config['diffusion_timesteps']
        if args.mask == 'random':
            Q_prod, Q_t = tokenizer.q_random_schedule(timesteps=diffusion_timesteps)
        if args.mask == 'blosum':
            Q_prod, Q_t = tokenizer.q_blosum_schedule(timesteps=diffusion_timesteps)
    masking_idx = tokenizer.mask_id
    padding_idx = tokenizer.pad_id
    print(n_tokens)
    print(masking_idx, padding_idx)

    # model = ByteNetLM(n_tokens, d_embed, d_model, n_layers, kernel_size, r,
    #                   causal=causal, padding_idx=masking_idx, rank=weight_rank, dropout=args.dropout,
    #                   tie_weights=args.tie_weights, final_ln=args.final_norm, slim=slim, activation=activation)
    model = ByteNetLMTime(n_tokens, d_embed, d_model, n_layers, kernel_size, r,
                          causal=causal, padding_idx=masking_idx, rank=weight_rank, dropout=args.dropout,
                          tie_weights=args.tie_weights, final_ln=args.final_norm, slim=slim, activation=activation,
                          timesteps=diffusion_timesteps)

    if args.checkpoint is not None:
        last_epoch = args.checkpoint
    else:
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
    #print(list(msd.keys())[0:10])
    if args.mask == 'so':
        msd = {k.split('module.')[1]: v for k, v in msd.items()}
    else:
        msd = {k.split('module.')[0]: v for k,v in msd.items()}
    #print(list(msd.keys())[0:10], list(model.state_dict().keys())[0:10])
    model.load_state_dict(msd) # TODO: why is this not saving the same

    sequences = args.num_seqs
    # The following were calculated from uniref dataset #
    seq_lengths = [64, 166, 268, 369, 471, 573, 675, 776, 878, 980, 1081, 1183, 1285, 1386, 1488, 1590, 1692, 1793,
                   1895, 1997]
    freq = [1, 0.8879753340184995, 0.5765673175745119, 0.35765673175745116, 0.19013360739979446, 0.131551901336074, \
            0.0698869475847893, 0.06063720452209661, 0.041109969167523124, 0.030832476875642344, 0.01644398766700925, \
            0.010277492291880781, 0.008221993833504625, 0.004513874614594039, 0.0067831449126413155, \
            0.009660842754367934, 0.003083247687564234, 0.0018633093525179856, 0.0043031860226104834, \
            0.02158273381294964]
    ###
    samples = [int(f * sequences) for f in freq]
    print(len(samples))
    print(len(seq_lengths))
    seqs = ""
    with open(args.out_fpath + 'generated_samples_string.csv', 'a') as f:
        for i, seq_len in enumerate(seq_lengths):
            sequences = samples[i]
            if (samples[i] != 0):
                for j in range(sequences):
                    print("sample", j)
                    if args.mask == 'autoreg' or args.mask=='so':
                        sample, string = generate_text(model, seq_len, tokenizer=tokenizer, penalty=args.penalty, causal=causal)
                        seqs += str(string)
                        seqs += "\n"
                    elif args.mask == 'blosum' or args.mask == 'random':
                        sample, string = generate_text_d3pm(model, seq_len, Q_bar=Q_prod, Q=Q_t,
                                                            timesteps=diffusion_timesteps, no_step=args.no_step)
                        seqs += str(string)
                        seqs += "\n"
                    f.write(str(string) + "\n")
        f.close()

    #np.savetxt(args.out_fpath + 'generated_samples.csv', seqs_array, delimiter=',', fmt='%d') # not necessary
    with open(args.out_fpath + 'generated_samples_string.csv', 'a') as f:
        f.write(','.join(
            [seqs]))
        f.write('\n')

def generate_text(model, seq_len, tokenizer=Tokenizer(), penalty=None, causal=False):
    # Generate a random start string and convert to tokens
    all_aas = tokenizer.all_aas
    mask = tokenizer.mask_id
    # Start from mask
    sample = torch.zeros((seq_len))+mask
    sample = sample.to(torch.long)
    print("input seq", tokenizer.untokenize(sample))
    # Unmask 1 loc at a time randomly
    loc = np.arange(seq_len)
    print(len(loc))
    if causal == False:
        np.random.shuffle(loc)
    with torch.no_grad():
        for i in loc:
            timestep = torch.tensor([0]) # placeholder not called
            prediction = model(sample.unsqueeze(0), timestep) #, input_mask=input_mask.unsqueeze(-1)) #sample prediction given input
            p = prediction[:, i, :len(all_aas)-6] # sample at location i (random), dont let it predict non-standard AA
            p = torch.nn.functional.softmax(p, dim=1) # softmax over categorical probs
            p_sample = torch.multinomial(p, num_samples=1)
            # Repetition penalty
            if penalty is not None: # ignore if value is None
                case1 = (i == 0 and sample[i+1] == p_sample) # beginning of seq
                case2 = (i == seq_len-1 and sample[i-1] == p_sample) # end of seq
                case3 = ((i < seq_len-1 and i > 0) and ((sample[i-1] == p_sample) or (sample[i+1] == p_sample))) # middle of seq
                if case1 or case2 or case3:
                    #print("identified repeat", p_sample, sample[i-1], sample[i+1])
                    p[:, int(p_sample)] /= penalty
                    p_sample = torch.multinomial(p, num_samples=1) # resample
            sample[i] = p_sample
            #print(tokenizer.untokenize(sample)) # check that sampling correctly
    print("final seq", tokenizer.untokenize(sample))
    return sample, tokenizer.untokenize(sample)

def generate_text_d3pm(model, seq_len, Q_bar=None, Q=None, tokenizer=Tokenizer(), timesteps=500, no_step=False):
    """
    no_step: if true will calculate p_tilde(x_0|x_t)
             if false will calculate p_tilde(x_tminus1|x_t) for each t in timestep
    """
    # Generate a random start string from uniform dist and convert to tokens
    all_aas = tokenizer.all_aas
    sample = torch.randint(0, len(all_aas), (seq_len,))
    sample = sample.to(torch.long)
    print("input seq", tokenizer.untokenize(sample))
    if no_step:
        timesteps = torch.linspace(timesteps-1, timesteps-1, steps=1, dtype=torch.long)
    else:
        timesteps = torch.linspace(timesteps-1,1,timesteps-1) # iterate over reverse timesteps
        timesteps = timesteps.to(torch.long)
    with torch.no_grad():
        for t in timesteps:
            x_t = tokenizer.one_hot(sample)
            prediction = model(sample.unsqueeze(0), t.unsqueeze(0))
            p = prediction[:, :, :len(all_aas)]  # p_theta_tilde (x_0_tilde | x_t)
            p = torch.nn.functional.softmax(p, dim=2).squeeze()  # softmax over categorical probs
            p = p.to(torch.float64)
            if no_step:
                x_tminus1 = torch.multinomial(p, num_samples=1).squeeze()
            else:
                # Calculate p_theta_marg from p_theta_tilde
                A = torch.mm(x_t, torch.t(Q[t]))  # [P x K]
                B = Q_bar[t - 1] # [K, K]
                q_t = torch.mul(A.unsqueeze(1), B)  # [ P x K x K ]
                p_theta_marg = torch.bmm(q_t, p.unsqueeze(2)).squeeze()  # this marginalizes over dim=2
                p_theta_marg = p_theta_marg / p_theta_marg.sum(axis=1, keepdim=True)
                x_tminus1 = torch.multinomial(p_theta_marg, num_samples=1).squeeze()
            diff = torch.ne(sample, x_tminus1)
            if t % 50 == 0:
                print("time", t, diff.sum().item(), "mutations", tokenizer.untokenize(x_tminus1))
            sample = x_tminus1
    print("final seq", tokenizer.untokenize(sample))
    return sample, tokenizer.untokenize(sample)

if __name__ == '__main__':
    main()
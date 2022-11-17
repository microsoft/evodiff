from dms.model import ByteNetLMTime
import numpy as np
import argparse
from sequence_models.constants import MSA_ALPHABET, ALL_AAS, PROTEIN_ALPHABET, PAD
import torch
import os
import json
from dms.utils import Tokenizer
import pathlib
from tqdm import tqdm

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

    _ = torch.manual_seed(0)
    np.random.seed(0)

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

    if args.mask == 'autoreg' or args.mask == 'so' or args.mask == 'reference':
        tokenizer = Tokenizer()
        diffusion_timesteps = None  # Not input to model
        if args.mask == 'so':
            n_tokens = len(PROTEIN_ALPHABET)
            tokenizer = Tokenizer(protein_alphabet=PROTEIN_ALPHABET, all_aas=ALL_AAS, pad=PAD)
            causal = True
    elif args.mask == 'blosum' or args.mask == 'random':
        tokenizer = Tokenizer(path_to_blosum=data_top_dir + "blosum62-special-MSA.mat", sequences=True)
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
                          timesteps=diffusion_timesteps) # works w/ time and non-time models (when diffusion_timesteps is None)

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

    if args.mask != 'reference':
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
        model.load_state_dict(msd)

    sequences = args.num_seqs
    seq_lengths = [32, 64, 128, 256, 384, 512] #, 1024, 2048]
    seqs = ""

    for i, seq_len in enumerate(seq_lengths):
        with open(args.out_fpath + 'generated_samples_string_'+str(seq_len)+'.fasta', 'a') as f:
            count = 0
            fasta_string = ""

            if args.mask == 'autoreg' or args.mask=='so':
                sample, string = generate_text(model, seq_len, tokenizer=tokenizer, penalty=args.penalty, causal=causal,
                                               batch_size=args.num_seqs)
            elif args.mask == 'blosum' or args.mask == 'random':
                sample, string = generate_text_d3pm(model, seq_len, Q_bar=Q_prod, Q=Q_t, tokenizer=tokenizer,
                                                    timesteps=diffusion_timesteps, no_step=args.no_step,
                                                    batch_size=args.num_seqs)
            elif args.mask == 'reference':
                sample, string = generate_random_seq(seq_len, tokenizer=tokenizer)

            for _s in string:
                fasta_string += ">SEQUENCE_" + str(count) + "\n" + str(_s) + "\n"
                count += 1
                seqs += str(_s) + "\n"

            f.write(fasta_string)
            f.close()

    with open(args.out_fpath + 'generated_samples_string.csv', 'a') as f:
        f.write(','.join(
            [seqs]))
        f.write('\n')

def generate_text(model, seq_len, tokenizer=Tokenizer(), penalty=None, causal=False, batch_size=20):
    # Generate a random start string and convert to tokens
    all_aas = tokenizer.all_aas
    mask = tokenizer.mask_id
    # Start from mask
    sample = torch.zeros((batch_size, seq_len))+mask
    sample = sample.to(torch.long)
    #print("input seq", tokenizer.untokenize(sample))
    # Unmask 1 loc at a time randomly
    loc = np.arange(seq_len)
    if causal == False:
        np.random.shuffle(loc)
    with torch.no_grad():
        for i in loc:
            timestep = torch.tensor([0] * batch_size) # placeholder but not called in model
            prediction = model(sample, timestep) #, input_mask=input_mask.unsqueeze(-1)) #sample prediction given input
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
            sample[:, i] = p_sample.squeeze()
            #print([tokenizer.untokenize(s) for s in sample]) # check that sampling correctly

    print("final seq", [tokenizer.untokenize(s) for s in sample])
    untokenized = [tokenizer.untokenize(s) for s in sample]
    return sample, untokenized

def generate_text_d3pm(model, seq_len, Q_bar=None, Q=None, tokenizer=Tokenizer(), timesteps=500, no_step=False,
                       batch_size=20):
    """
    no_step: if true will calculate p_tilde(x_0|x_t) from a uniform sample
             if false will calculate p_tilde(x_tminus1|x_t) for each t in timestep
    """
    # Generate a random start string from uniform dist and convert to tokens
    #all_aas = tokenizer.all_aas

    sample = torch.randint(0, tokenizer.K, (batch_size, seq_len)) # don't include gap token?
    sample = sample.to(torch.long)
    #print("input seq", tokenizer.untokenize(sample))
    if no_step:
        timesteps = np.linspace(timesteps-1, timesteps-1, steps=1, dtype=int)
    else:
        timesteps = np.linspace(timesteps-1,1,int((timesteps-1)/1), dtype=int) # iterate over reverse timesteps
    with torch.no_grad():
        for t in timesteps:
            timesteps = [t] * batch_size
            prediction = model(sample, torch.tensor(timesteps))
            p = prediction[:, :, :tokenizer.K]  # p_theta_tilde (x_0_tilde | x_t)
            p = torch.nn.functional.softmax(p, dim=2).squeeze()  # softmax over categorical probs
            p = p.to(torch.float64)
            if no_step:
                x_tminus1 = torch.multinomial(p, num_samples=1)
            else:
                x_tminus1 = sample.clone()
                for i, s in enumerate(sample):
                    # Calculate p_theta_marg from p_theta_tilde
                    x_t_b = tokenizer.one_hot(s)
                    A = torch.mm(x_t_b, torch.t(Q[t]))  # [P x K]
                    Q_expand = Q_bar[t-1].unsqueeze(0).expand(A.shape[0], tokenizer.K, tokenizer.K)  # [ P x K x K]
                    B_pred = torch.mul(p[i].unsqueeze(2), Q_expand)
                    q_t = torch.mul(A.unsqueeze(1), B_pred)  # [ P x K x K ]
                    p_theta_marg = torch.bmm(q_t,  p[i].unsqueeze(2)).squeeze()  # this marginalizes over dim=2
                    p_theta_marg = p_theta_marg / p_theta_marg.sum(axis=1, keepdim=True)
                    x_tminus1[i] = torch.multinomial(p_theta_marg, num_samples=1).squeeze()
                    diff = torch.ne(s, x_tminus1[i])
                    if t % 20 == 0:
                        print("time", t, diff.sum().item(), "mutations") #, tokenizer.untokenize(x_tminus1))
            sample = x_tminus1
    untokenized = [tokenizer.untokenize(s) for s in sample]
    print("final seq", untokenized)
    return sample, untokenized

def generate_random_seq(seq_len, tokenizer=Tokenizer()):
    """
    Generates a set of random sequences drawn from a uniform distribution
    """
    # Generate a random start string from uniform dist and convert to tokens
    all_aas = tokenizer.all_aas
    print(all_aas)
    sample = torch.randint(0, len(all_aas)-4, (seq_len,)) # ignore char not accepted by PROSITE
    sample = sample.to(torch.long)
    print("sequence", tokenizer.untokenize(sample))
    return sample, tokenizer.untokenize(sample)

if __name__ == '__main__':
    main()
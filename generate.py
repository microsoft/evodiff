#from sequence_models.convolutional import ByteNetLM # TODO NEED TO USE BYTENETLMTIME for BLOSUM
from model import ByteNetLMTime
import numpy as np
import argparse
from dms.constants import ALL_AAS, MSA_PAD, PROTEIN_ALPHABET
from sequence_models.constants import MASK, PAD
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
    parser.add_argument('--final_norm', action='store_true') # TODO THIS SHOULD BE ON
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--num_seqs', type=int, default=100)
    parser.add_argument('--mask', type=str, default='autoreg')
    args = parser.parse_args()

    with open(args.config_fpath, 'r') as f:
        config = json.load(f)

    n_tokens = len(PROTEIN_ALPHABET) # TODO ADD START/STOP FOR KEVIN
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
    causal = False
    if args.mask == 'autoreg':
        tokenizer = Tokenizer()
        diffusion_timesteps = None  # Not input to model
    elif args.mask == 'blosum' or args.mask == 'random':
        tokenizer = Tokenizer(path_to_blosum=data_top_dir + "blosum62-special.mat")
        diffusion_timesteps = config['diffusion_timesteps']
    elif args.mask == 'so':
        causal=True
    masking_idx = tokenizer.pad_id

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
    msd = {k.split('module.')[0]: v for k,v in msd.items()} # TODO: this was zero (for OAARDM)
    #print(list(msd.keys())[0:10], list(model.state_dict().keys())[0:10])
    model.load_state_dict(msd) # TODO: why is this not saving the same

    sequences = args.num_seqs
    seqs = ""
    # The following were calculated from uniref dataset #
    seq_lengths = [150] #64, 166, 268, 369, 471, 573, 675, 776, 878, 980, 1081, 1183, 1285, 1386, 1488, 1590, 1692, 1793,
                   #1895, 1997]
    freq = [1.0] #, 0.8879753340184995, 0.5765673175745119, 0.35765673175745116, 0.19013360739979446, 0.131551901336074, \
            # 0.0698869475847893, 0.06063720452209661, 0.041109969167523124, 0.030832476875642344, 0.01644398766700925, \
            # 0.010277492291880781, 0.008221993833504625, 0.004513874614594039, 0.0067831449126413155,
            # 0.009660842754367934,
            # 0.003083247687564234, 0.0018633093525179856, 0.0043031860226104834, 0.02158273381294964]
    ###
    samples = [int(f * sequences) for f in freq]
    for i, seq_len in enumerate(seq_lengths):
        sequences = samples[i]
        for j in range(sequences):
            if args.mask == 'autoreg':
                sample, string = generate_text(model, seq_len, timesteps=None)
                seqs += str(string)
                seqs += "\n"
            elif args.mask == 'd3pm':
                sample, string = generate_text_d3pm(model, seq_len, timesteps=diffusion_timesteps)
                seqs += str(string)
                seqs += "\n"

    #np.savetxt(args.out_fpath + 'generated_samples.csv', seqs_array, delimiter=',', fmt='%d') # not necessary
    with open(args.out_fpath + 'generated_samples_string.csv', 'a') as f:
        f.write(','.join(
            [seqs]))
        f.write('\n')

def generate_text(model, seq_len, tokenizer=Tokenizer(), timesteps=None):
    # Generate a random start string and convert to tokens
    all_aas = tokenizer.tokenize([ALL_AAS])
    mask = tokenizer.tokenize(MASK)
    # Start from mask
    sample = torch.zeros((seq_len))+mask
    sample = sample.to(torch.long)
    print("input seq", tokenizer.untokenize(sample))
    # Unmask 1 loc at a time randomly
    loc = np.arange(seq_len)
    np.random.shuffle(loc)
    with torch.no_grad():
        for i in loc:
            #print("pos", i)
            #print(tokenizer.untokenize(sample))
            prediction = model(sample.unsqueeze(0), timesteps) #, input_mask=input_mask.unsqueeze(-1)) #sample prediction given input
            p = prediction[:, i, :len(all_aas)-6] # sample at location i (random), dont let it predict non-standard AA
            p = torch.nn.functional.softmax(p, dim=1) # softmax over categorical probs
            p_sample = torch.multinomial(p, num_samples=1)
            sample[i] = p_sample
    print("final seq", tokenizer.untokenize(sample))
    return sample, tokenizer.untokenize(sample)

def generate_text_d3pm(model, seq_len, tokenizer=Tokenizer(), timesteps=500):
    # Generate a random start string and convert to tokens
    all_aas = tokenizer.all_aas
    # Start from random array (not including non-standard AA (6))
    sample = torch.randint(0, len(all_aas)-6, (seq_len,))
    sample = sample.to(torch.long)
    print("input seq", tokenizer.untokenize(sample))
    timesteps = torch.linspace(timesteps-1,0,timesteps) # iterate over reverse timesteps
    timesteps = timesteps.to(torch.long)
    with torch.no_grad():
        for t in timesteps:
            prediction = model(sample.unsqueeze(0), t.unsqueeze(0)) #sample prediction given input
            p = prediction[:, :, :len(all_aas)-6] # sample all locations (random) # 6 = nonstandard AA
            p = torch.nn.functional.softmax(p, dim=1) # softmax over categorical probs
            p_sample = torch.multinomial(p, num_samples=1)
            sample = p_sample
            #print("sample i after", tokenizer.untokenize(sample))
    print("final seq", tokenizer.untokenize(sample[:seq_len]))
    return sample[:seq_len], tokenizer.untokenize(sample[:seq_len])

if __name__ == '__main__':
    main()
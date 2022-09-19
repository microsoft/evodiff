from sequence_models.convolutional import ByteNetLM # TODO NEED TO USE BYTENETLMTIME for BLOSUM
import numpy as np
import argparse
from dms.constants import ALL_AAS, MSA_PAD, PROTEIN_ALPHABET
from sequence_models.constants import MASK, PAD
import torch
import os
import json
from dms.utils import Tokenizer

### SET RANDOM SEEDS ####
random_seed = 1
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)
torch.cuda.empty_cache() # empty caches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_fpath')
    parser.add_argument('out_fpath', type=str, nargs='?', default=os.getenv('PT_OUTPUT_DIR', '/tmp') + '/')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--final_norm', action='store_true') # TODO THIS SHOULD BE ON
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--num_seqs', type=int, default=100)
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
    seq_len = args.seq_len
    causal = False
    tokenizer=Tokenizer()
    masking_idx = tokenizer.mask_id
    padding_idx = tokenizer.pad_id

    model = ByteNetLM(n_tokens, d_embed, d_model, n_layers, kernel_size, r,
                      causal=causal, padding_idx=padding_idx, rank=weight_rank, dropout=args.dropout,
                      tie_weights=args.tie_weights, final_ln=args.final_norm, slim=slim, activation=activation)

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
    seqs_array = np.zeros((sequences, 512))
    for i in range(sequences):
        sample, string = generate_text(model, seq_len)
        seqs += str(string)
        seqs += "\n"
        seqs_array[i] = sample.numpy()

    np.savetxt(args.out_fpath + 'generated_samples.csv', seqs_array, delimiter=',', fmt='%d')
    with open(args.out_fpath + 'generated_samples_string.csv', 'a') as f:
        f.write(','.join(
            [seqs]))
        f.write('\n')

def generate_text(model, seq_len, tokenizer=Tokenizer()):
    # Generate a random start string and convert to tokens
    #padding_idx = tokenizer.tokenize(MSA_PAD)[0]
    all_aas = tokenizer.tokenize([ALL_AAS])
    alphabet = tokenizer.tokenize([PROTEIN_ALPHABET])
    mask = tokenizer.tokenize(MASK)
    # Start from mask or random array
    sample = torch.zeros((seq_len))+mask
    sample = sample.to(torch.long)
    seq = tokenizer.untokenize(sample)
    print("input seq", seq)
    # Unmask 1 loc at a time randomly
    loc = np.arange(seq_len)
    np.random.shuffle(loc)
    input_mask = torch.zeros(len(seq), dtype=bool)
    #print(loc.dtype, input_mask.dtype)
    with torch.no_grad():
        for i in loc:
            input_mask[i] = 1
            prediction = model(sample.unsqueeze(0), input_mask=input_mask) #sample prediction given input
            p = prediction[:, i, :len(all_aas)-6] # sample at location i (random)
            p = torch.nn.functional.softmax(p, dim=1) # softmax over categorical probs
            p_sample = torch.multinomial(p.squeeze(), num_samples=1)
            #p_sample = alphabet[np.argmax(p[i])] # over samples alanine
            sample[i] = p_sample
            #print("residue", i, tokenizer.untokenize(sample))
    print("final seq", tokenizer.untokenize(sample))
    return sample, tokenizer.untokenize(sample)

if __name__ == '__main__':
    main()
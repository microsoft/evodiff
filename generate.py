from sequence_models.convolutional import ByteNetLM
import numpy as np
import argparse
from dms.constants import PAD, PROTEIN_ALPHABET, BLOSUM62_AAS
from sequence_models.constants import MASK
import torch
import os
import json
from dms.collaters import random_sample
from dms.utils import Tokenizer

### SET RANDOM SEEDS ###
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
    parser.add_argument('--final_norm', action='store_true')
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--checkpoint', type=int, default=None)
    args = parser.parse_args()

    with open(args.config_fpath, 'r') as f:
        config = json.load(f)

    n_tokens = len(PROTEIN_ALPHABET)
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
    padding_idx = PROTEIN_ALPHABET.index(PAD)

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
    msd = {k.split('module.')[0]: v for k,v in msd.items()}
    model.load_state_dict(msd) # TODO: why is this not saving the same

    sequences = 100
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
    padding_idx = tokenizer.tokenize(PAD)[0]
    all_aas = tokenizer.tokenize([BLOSUM62_AAS])
    alphabet = tokenizer.tokenize([PROTEIN_ALPHABET])
    mask = tokenizer.tokenize(MASK)
    # Start from mask or random array
    sample = torch.zeros((1,512))+mask
    sample = sample.to(torch.long)
    seq = tokenizer.untokenize(sample[0])
    print("input seq", seq)
    # Unmask 1 loc at a time randomly
    loc = np.arange(seq_len)
    np.random.shuffle(loc)
    input_mask = torch.zeros(len(seq), dtype=bool)
    #print(loc.dtype, input_mask.dtype)
    for x,i in enumerate(loc):
        input_mask[i] = 1
        prediction = model(sample, input_mask=input_mask)
        p = torch.nn.functional.softmax(prediction[0], dim=1).detach().numpy()
        p_sample = np.random.choice(alphabet, p=p[i])
        #p_sample = alphabet[np.argmax(p[i])] # over samples alanine
        sample[0][i] = p_sample
        #print(x, i, sample)
    print(tokenizer.untokenize(sample[0]))
    return sample[0], tokenizer.untokenize(sample[0])

if __name__ == '__main__':
    main()
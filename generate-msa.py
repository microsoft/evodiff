import argparse
import json
import os
import numpy as np
import torch
import pandas as pd
from sequence_models.esm import MSATransformer
from sequence_models.constants import MSA_ALPHABET, MSA_PAD, MASK
from evodiff.utils import Tokenizer
from sequence_models.utils import parse_fasta
from evodiff.model import MSATransformerTime
from evodiff.data import read_idr_files
from tqdm import tqdm
import pathlib
import glob
import string

from evodiff.data import A3MMSADataset, IDRDataset
from torch.utils.data import Subset
from sequence_models.samplers import SortishSampler, ApproxBatchSampler
from torch.utils.data import DataLoader
import torch
from sequence_models.collaters import MSAAbsorbingCollater
from evodiff.collaters import D3PMCollaterMSA
from sequence_models.constants import MSA_ALPHABET
from evodiff.utils import Tokenizer
from scipy.spatial.distance import hamming, cdist

home = str(pathlib.Path.home())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_fpath')
    parser.add_argument('out_fpath', type=str, nargs='?',
                        default=os.getenv('AMLT_OUTPUT_DIR', '/tmp') + '/')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-off', '--offset', default=0, type=int,
                        help='Number of GPU devices to skip.')
    parser.add_argument('-sd', '--state_dict', default=None)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--final_norm', action='store_true')
    parser.add_argument('--mask', type=str, default='autoreg')
    parser.add_argument('--batch-size', type=int, default=20) # batch-size (on amlt use 1)
    parser.add_argument('--n-sequences', type=int, default=64)
    parser.add_argument('--seq-length', type=int, default=512)
    parser.add_argument('--penalty-value', type=float, default=0) # Default no penalty /=1 on gap generation
    parser.add_argument('--subsampling', type=str, default='MaxHamming')
    parser.add_argument('--delete-prev', action='store_true')  # Will delete previous generated sequences that start with generated* in main folder
    parser.add_argument('--start-query', action='store_true') # if starting from query -> gen msa
    parser.add_argument('--start-msa', action='store_true') # if starting from msa, gen-> query
    parser.add_argument('--idr', action='store_true') # if doing idr generation
    parser.add_argument('--amlt', action='store_true') # if running on amlt
    parser.add_argument('--run', type=int, default=0) # for conditional generation of idrs to query data
    args = parser.parse_args()

    #_ = torch.manual_seed(0)
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
        data_top_dir = os.getenv('AMLT_DATA_DIR') + '/data/data/data/' # TODO i messed up my amulet storage - this works for now
        data_dir = data_top_dir
        data_dir += config['dataset'] + '/'
        ptjob = True
    except:
        data_top_dir = 'data/'
        #print(data_top_dir)
        data_dir = data_top_dir
        data_dir += config['dataset'] + '/'
        ptjob = False

    project_dir = home + '/Desktop/DMs/'

    if args.start_query and args.start_msa:
        raise Exception("Can only choose either start-query or start-msa NOT both, to generate from scratch omit flags")

    if args.delete_prev:
        filelist = glob.glob(args.out_fpath+'generated*')
        filelist += glob.glob(args.out_fpath+'msas/*generated*')
        filelist += glob.glob(args.out_fpath+'valid*')
        for file in filelist:
            os.remove(file)
            print("Deleting", file)
    if args.penalty_value > 0:
        print("Penalizing GAPS by factor of", 1+args.penalty_value)
    else:
        print("Not penalizing GAPS")
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
        model = MSATransformer(d_embed, d_hidden, n_layers, n_heads, use_ckpt=True, n_tokens=len(MSA_ALPHABET),
                               padding_idx=MSA_ALPHABET.index(MSA_PAD), mask_idx=MSA_ALPHABET.index(MASK)).cuda()
    else:
        model = MSATransformerTime(d_embed, d_hidden, n_layers, n_heads, timesteps=diffusion_timesteps, use_ckpt=True,
                                   n_tokens=len(MSA_ALPHABET), padding_idx=padding_idx, mask_idx=masking_idx).cuda()

    model = model.to(device)

    # Restore the model weights for the last checkpoint after training
    if args.amlt: # For generating on SINGULARITY
        print(os.getenv('AMLT_DATA_DIR'))
        print(args.out_fpath)
        args.out_fpath = os.getenv('AMLT_DATA_DIR') + '/checkpoints/' + args.out_fpath # checkpoints are located on amlt storage data/checkpoints/ + job_name (e.g. job_name = diff/oaardm_msa_maxham)
        print(args.out_fpath)
    outputs = os.listdir(args.out_fpath)
    if len(outputs) > 0:
       last_epoch = 0
       for output in outputs:
           if 'checkpoint' in output:
               print(output)
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

    if args.amlt:
        args.out_fpath = os.getenv('AMLT_OUTPUT_DIR', '/tmp') + '/'

    if args.idr:
        sample, _string, original_idr, new_idr = generate_idr(model, tokenizer, args.n_sequences, args.seq_length,
                                      index=args.run, penalty_value=args.penalty_value, device=device, start_query=args.start_query,
                                      data_top_dir=data_top_dir, selection_type=args.subsampling, out_path=args.out_fpath)

        # if not os.path.exists(args.out_fpath + 'idrs/'):
        #     os.makedirs(args.out_fpath + 'idrs/')
        for count, msa in enumerate(_string):
            fasta_string = ""
            with open(args.out_fpath + 'generated_msas.a3m', 'a') as f:
                for seq in range(args.n_sequences):
                    seq_num = seq * args.seq_length
                    next_seq_num = (seq+1) * args.seq_length
                    seq_string = str(msa[0][seq_num:next_seq_num]).replace('!', '')  # remove PADs
                    if seq_num == 0 :
                        f.write(">SEQUENCE_0" + "\n" + str(seq_string) + "\n")
                    else:
                        f.write(">tr \n" + str(seq_string) + "\n" )
                f.write(fasta_string)
                f.close()
    else:
        if args.mask == 'autoreg':
            sample, _string = generate_msa(model, tokenizer, args.batch_size, args.n_sequences, args.seq_length,
                                          penalty_value=args.penalty_value, device=device, start_query=args.start_query,
                                           start_msa=args.start_msa,
                                          data_top_dir=data_top_dir, selection_type=args.subsampling, out_path=args.out_fpath)
        elif args.mask == 'blosum' or args.mask=='random':
            sample, _string = generate_msa_d3pm(model, args.batch_size, args.n_sequences, args.seq_length,
                                               Q_bar=Q_prod, Q=Q_t, tokenizer=Tokenizer(), data_top_dir=data_top_dir,
                                               selection_type=args.subsampling, out_path=args.out_fpath,
                                               max_timesteps=diffusion_timesteps, start_query=args.start_query,
                                               no_step=False, penalty_value=args.penalty_value, device=device)

        # Save strings to a3m; save each MSA to new file
        # if not os.path.exists(args.out_fpath + 'msas/'):
        #     os.makedirs(args.out_fpath + 'msas/')
        for count, msa in enumerate(_string):
            fasta_string = ""
            #count_str = "".join(np.random.choice([*string.ascii_uppercase], size=3, replace=True)) # randomly assign new name
            with open(args.out_fpath + 'generated_msas.a3m', 'a') as f:
                for seq in range(args.n_sequences):
                    seq_num = seq * args.seq_length
                    next_seq_num = (seq+1) * args.seq_length
                    seq_string = str(msa[0][seq_num:next_seq_num]).replace('!', '')  # remove PADs
                    if seq_num == 0 :
                        f.write(">MSA_0" + "\n" + str(seq_string) + "\n")
                    else:
                        f.write(">tr \n" + str(seq_string) + "\n" )
                f.write(fasta_string)
                f.close()

        # cat all files to one file
        # msafilelist = glob.glob(args.out_fpath + 'msas/*generated*')
        # with open(args.out_fpath+'generated_msas.a3m', 'a') as f:
        #     for fname in msafilelist:
        #         with open(fname) as infile:
        #             for line in infile:
        #                 f.write(line)

        # Save tokenized seqs to npz file
        np.save(args.out_fpath+'generated_msas', np.array(sample.cpu()))


def generate_msa(model, tokenizer, batch_size, n_sequences, seq_length, penalty_value=2, device='gpu',
                 start_query=False, start_msa=False, data_top_dir='../data', selection_type='MaxHamming', out_path='../ref/'):
    mask_id = tokenizer.mask_id
    src = torch.full((batch_size, n_sequences, seq_length), fill_value=mask_id)
    masked_loc_x = np.arange(n_sequences)
    masked_loc_y = np.arange(seq_length)
    if start_query:
        valid_msas, query_sequences, tokenizer =get_valid_data(data_top_dir, batch_size, 'autoreg', data_dir='openfold/',
                                       selection_type=selection_type, n_sequences=n_sequences, max_seq_len=seq_length,
                                       out_path=out_path)
        # First row is query sequence
        for i in range(batch_size):
            #print(len(query_sequences))
            #import pdb; pdb.set_trace()
            seq_len = len(query_sequences[i])
            print("PAD ID", tokenizer.pad_id)
            src[i][0][:seq_len] = query_sequences[i]
            padding = torch.full((n_sequences, seq_length-seq_len), fill_value=tokenizer.pad_id)
            # print(query_sequences[i].shape)
            # print(padding.shape)
            # import pdb; pdb.set_trace()
            src[i,:,seq_len:] = padding
            x_indices = np.arange(1,n_sequences)
            y_indices = np.arange(seq_len)
    elif start_msa:
        valid_msas, query_sequences, tokenizer = get_valid_data(data_top_dir, batch_size, 'autoreg',
                                                                data_dir='openfold/',
                                                                selection_type=selection_type, n_sequences=n_sequences,
                                                                max_seq_len=seq_length,
                                                                out_path=out_path)
        for i in range(batch_size):
            seq_len = len(query_sequences[i])
            src[i, 1:n_sequences, :seq_len] = valid_msas[i][0, 1:n_sequences, :seq_len].squeeze()
            padding = torch.full((n_sequences, seq_length-seq_len), fill_value=tokenizer.pad_id)
            src[i, :, seq_len:] = padding
            x_indices = np.arange(0,1)
            y_indices = np.arange(seq_len)
    src = src.to(device)
    sample = src.clone()
    if start_query or start_msa:
        all_ind = np.transpose([np.tile(x_indices, len(y_indices)), np.repeat(y_indices, len(x_indices))])
    else:
        all_ind = np.transpose([np.tile(masked_loc_x, len(masked_loc_y)), np.repeat(masked_loc_y, len(masked_loc_x))])
    np.random.shuffle(all_ind)

    with torch.no_grad():
        #all_ind = all_ind[:10] for debugging TODO delte
        for i in tqdm(all_ind):
            random_x, random_y = i
            preds = model(sample)  # Output shape of preds is (BS=1, N=64, L, n_tokens=31)
            p = preds[:, random_x, random_y, :]
            if random_x == 0 : # for first row don't let p_softmax predict gaps
                p = preds[:, random_x, random_y, :tokenizer.K-1]
            p_softmax = torch.nn.functional.softmax(p, dim=1)
            # Penalize gaps
            penalty = torch.ones(p.shape).to(p.device)
            penalty[:, -1] += penalty_value
            #print(p_softmax)
            p_softmax /= penalty
            #print(p_softmax)
            p_sample = torch.multinomial(input=p_softmax, num_samples=1)
            p_sample = p_sample.squeeze()
            # for b in range(batch_size): # Iterate over batches and only replace correct res in query (ignore padding)
            #     if start_query and random_x not in x_indices[b] or random_y not in y_indices[b]:
            #         pass
            #     else:
            #        sample[b, random_x, random_y] = p_sample
            sample[:, random_x, random_y] = p_sample
            print("time", random_x, random_y, "sample", tokenizer.untokenize(sample[0].flatten()))
    untokenized = [[tokenizer.untokenize(msa.flatten())] for msa in sample]
    return sample, untokenized # return output and untokenized output


# def generate_idr(model, tokenizer, n_sequences, seq_length, penalty_value=2, device='gpu', index=0,
#                  start_query=False, data_top_dir='../data', selection_type='MaxHamming', out_path='../ref/'):
#     src, start_idx, end_idx, original_idr, num_sequences = get_IDR_MSAs(index, data_top_dir, tokenizer, max_seq_len=seq_length,
#                                                          n_sequences=n_sequences, out_path=out_path,
#                                                          selection_type=selection_type)
#     src = torch.tensor(src).unsqueeze(0) # Make batchsize 1
#
#     masked_loc_x = np.arange(num_sequences) # len of MSA ; num sequences
#     masked_loc_y = np.arange(start_idx, end_idx)
#     all_ind = np.transpose([np.tile(masked_loc_x, len(masked_loc_y)), np.repeat(masked_loc_y, len(masked_loc_x))])
#     np.random.shuffle(all_ind)
#
#     sample = src.clone()
#     sample = sample.to(device)
#
#     with torch.no_grad():
#         for i in tqdm(all_ind):
#             #print(i)
#             random_x, random_y = i
#             print(sample.shape)
#             preds = model(sample)  # Output shape of preds is (BS=1, N=64, L, n_tokens=31)
#             print("preds", preds.shape)
#             print(random_x, random_y)
#             p = preds[:, random_x, random_y, :]
#             # if random_x == 0:  # for first row don't let p_softmax predict gaps
#             #     p = preds[:, random_x, random_y, :tokenizer.K - 1]
#             p_softmax = torch.nn.functional.softmax(p, dim=1)
#             # Penalize gaps
#             penalty = torch.ones(p.shape).to(p.device)
#             penalty[:, -1] += penalty_value
#             # print(p_softmax)
#             p_softmax /= penalty
#             # print(p_softmax)
#             p_sample = torch.multinomial(input=p_softmax, num_samples=1)
#             p_sample = p_sample.squeeze()
#             sample[:, random_x, random_y] = p_sample
#             print(tokenizer.untokenize(sample[0][0]))
#     print(sample.shape)
#     #print([tokenizer.untokenize(seq) for seq in sample[0]])
#     new_idr = [tokenizer.untokenize(seq[start_idx:end_idx]) for seq in sample[0]]
#     untokenized = [[tokenizer.untokenize(msa.flatten())] for msa in sample[0]]
#
#     #print(untokenized[0])
#     return sample, untokenized, original_idr, new_idr  # return output and untokenized output


def generate_msa_d3pm(model, batch_size, n_sequences, seq_length, Q_bar=None, Q=None, tokenizer=Tokenizer(),
                      start_query=False, data_top_dir='../data', selection_type='MaxHamming', out_path='../ref/',
                      max_timesteps=500, no_step=False, penalty_value=0, device='gpu'):
    sample = torch.randint(0, tokenizer.K, (batch_size, n_sequences, seq_length))
    #sample[:, 0, :] = torch.randint(0, tokenizer.K-1, (batch_size, 1, seq_length)) # resample query seq with no gaps
    if start_query:
        x_indices = []
        y_indices = []
        valid_msas, query_sequences, tokenizer =get_valid_data(data_top_dir, batch_size, 'autoreg', data_dir='openfold/',
                                       selection_type=selection_type, n_sequences=n_sequences, max_seq_len=seq_length,
                                       out_path=out_path)
        # First row is query sequence
        for i in range(batch_size):
            seq_len = len(query_sequences[i])
            print("PAD ID", tokenizer.pad_id)
            sample[i][0][:seq_len] = query_sequences[i]
            padding = torch.full((n_sequences, seq_length-seq_len), fill_value=tokenizer.pad_id)
            sample[i,:,seq_len:] = padding
            x_indices.append(np.arange(1,n_sequences))
            y_indices.append(np.arange(seq_length-seq_len))
    sample = sample.to(torch.long)
    sample = sample.to(device)
    [print("input query seq", tokenizer.untokenize(sample[i].flatten()[:seq_length])) for i in range(batch_size)]
    #import pdb; pdb.set_trace()
    print(sample.shape)
    if no_step:
        timesteps = np.linspace(max_timesteps-1, max_timesteps-1, 1, dtype=int)
    else:
        timesteps = np.linspace(max_timesteps-1,1,int((max_timesteps-1)/1), dtype=int) # iterate over reverse timesteps
    with torch.no_grad():
        print(timesteps[-1])
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
                    # FIRST UNPAD sample in batch
                    if start_query:
                        s = s[:, :len(y_indices[i])]
                        p_current = p[i, :, :len(y_indices[i])].flatten(start_dim=0, end_dim=1)
                    else:
                        p_current = p[i].flatten(start_dim=0, end_dim=1)
                    x_t_b = torch.stack([tokenizer.one_hot(s_i) for s_i in s])
                    x_t_b = x_t_b.flatten(start_dim=0, end_dim=1)
                    #p_current = p[i].flatten(start_dim=0, end_dim=1)
                    A = torch.mm(x_t_b, torch.t(Q[t]))  # [P x K]
                    Q_expand = Q_bar[t-1].unsqueeze(0).expand(A.shape[0], tokenizer.K, tokenizer.K)  # [ P x K x K]
                    B_pred = torch.mul(p_current.unsqueeze(2), Q_expand)
                    q_t = torch.mul(A.unsqueeze(1), B_pred)  # [ P x K x K ]
                    p_theta_marg = torch.bmm(torch.transpose(q_t, 1,2),  p_current.unsqueeze(2)).squeeze()  # this marginalizes over dim=2
                    p_theta_marg = p_theta_marg / p_theta_marg.sum(axis=1, keepdim=True)
                    #print('before', p_theta_marg[:seq_length])
                    # Penalize gaps
                    #print(p_theta_marg.shape)
                    #import pdb; pdb.set_trace()
                    penalty = torch.ones(p_theta_marg.shape).to(p_theta_marg.device)
                    penalty[:, -1] += penalty_value
                    p_theta_marg /= penalty
                    x_tminus1_temp = torch.multinomial(p_theta_marg[:, :], num_samples=1).squeeze()
                    x_tminus1_temp[:seq_length] = torch.multinomial(p_theta_marg[:seq_length,:-1], num_samples=1).squeeze() # NO GAPS in query
                    # On final timestep pick next best from GAP prediction for query sequence
                    # if t == 1:
                    #     x_tminus1_temp[:seq_length] = torch.multinomial(p_theta_marg[:seq_length, :tokenizer.K-1], num_samples=1).squeeze()
                if start_query:
                    print(x_tminus1_temp.shape)
                    print(x_tminus1_temp)
                    print(x_tminus1.shape)
                    x_tminus1[i, 1:, :len(y_indices[i])] = x_tminus1_temp.reshape(-1, len(y_indices[i]))[1:, :]
                else:
                    x_tminus1[i] = x_tminus1_temp.reshape(n_sequences, seq_length)
                sample = x_tminus1
                # #Uncomment to track generation
                if t % 50 == 0:
                  #print("time", t, diff.sum().item(), "mutations") #, tokenizer.untokenize(x_tminus1))
                  print("time",t, tokenizer.untokenize(sample[0].flatten()[seq_length:seq_length*5]))
                  #print("time",t, tokenizer.untokenize(sample[1].flatten()[:seq_length*2]))
    untokenized = [[tokenizer.untokenize(sample[i].flatten())] for i in range(batch_size)]
    # print(len(untokenized[0]))
    # print(len(untokenized[0][0]))
    # print("final seq", untokenized[0][0][:seq_length*2])
    return sample, untokenized


def get_valid_data(data_top_dir, num_seqs, arg_mask, data_dir='openfold/', selection_type='MaxHamming', n_sequences=64, max_seq_len=512,
                   out_path='../DMs/ref/'):
    #start_valid = '_valid'
    valid_msas = []
    query_msas = []
    seq_lens = []

    _ = torch.manual_seed(1) # same seeds as training
    np.random.seed(1)

    dataset = A3MMSADataset(selection_type, n_sequences, max_seq_len, data_dir=os.path.join(data_top_dir,data_dir), min_depth=64)

    train_size = len(dataset)
    random_ind = np.random.choice(train_size, size=(train_size - 10000), replace=False)
    val_ind = np.delete(np.arange(train_size), random_ind)


    ds_valid = Subset(dataset, val_ind)

    if arg_mask == 'autoreg':
        tokenizer = Tokenizer()
        collater = MSAAbsorbingCollater(alphabet=MSA_ALPHABET)
    elif arg_mask == 'blosum' or arg_mask == 'random':
        diffusion_timesteps = 500
        tokenizer = Tokenizer(path_to_blosum=data_top_dir + "blosum62-special-MSA.mat")
        if arg_mask == 'random':
            Q_prod, Q_t = tokenizer.q_random_schedule(timesteps=diffusion_timesteps)
        if arg_mask == 'blosum':
            Q_prod, Q_t = tokenizer.q_blosum_schedule(timesteps=diffusion_timesteps)
        collater = D3PMCollaterMSA(tokenizer=tokenizer, num_timesteps=diffusion_timesteps, Q=Q_t, Q_bar=Q_prod)

    torch.seed()  # reset seed ater val_ind
    loader = DataLoader(dataset=ds_valid,
                        batch_size=1,
                        shuffle=True,
                        collate_fn=collater,
                        num_workers=8)

    count = 0
    #num_seqs = len(val_ind)
    print("NUM SEQS", num_seqs)
    for batch in tqdm(loader):
        if arg_mask == 'blosum' or arg_mask == 'random':
            src, src_one_hot, timestep, tgt, tgt_one_hot, Q, Q_prod, q = batch
        else:
            src, tgt, mask = batch
        if count < num_seqs:
            valid_msas.append(tgt)
            print("QUERY", tokenizer.untokenize(tgt[0][0]), tgt[0][0].shape)
            seq_lens.append(len(tgt[0][0]))
            query_msas.append(tgt[0][0])  # first sequence in batchsize=1
            count += len(tgt)
        else:
            break
    print("LEN VALID MSAS", len(valid_msas))
    untokenized = [[tokenizer.untokenize(msa.flatten())] for msa in valid_msas]
    fasta_string = ""
    with open(out_path + 'valid_msas.a3m', 'a') as f:
        for i, msa in enumerate(untokenized):
            for seq in range(n_sequences):
                seq_num = seq * seq_lens[i]
                next_seq_num = (seq+1) * seq_lens[i]
                if seq_num == 0 :
                    f.write(">SEQUENCE_" + str(i) + "\n" + str(msa[0][seq_num:next_seq_num]) + "\n")
                else:
                    f.write(">tr \n" + str(msa[0][seq_num:next_seq_num]) + "\n" )
        f.write(fasta_string)
        f.close()
    #np.save(out_path + arg_mask + selection_type + start_valid + '_tokenized_openfold_train_msas', np.array(train_msas))

    return valid_msas, query_msas, tokenizer


def reindex_IDR(start_idx, end_idx, query_seq, gapped_query_seq):
    old_idx = list(np.arange(1, len(query_seq) + 1))  # This starts at 1 and is inclusive
    gap_count = 0
    offset = []  # This tracks how many gaps between letters
    for aa in list(gapped_query_seq):
        if aa == '-':
            gap_count += 1
        else:
            offset.append(gap_count)

    assert len(offset) == len(old_idx)

    # Gen index in list corresponding to start_index
    old_start = old_idx.index(start_idx)
    old_end = old_idx.index(end_idx)

    # Add gaps to old index to get new start/end index
    new_start = offset[old_start] + start_idx
    new_end = offset[old_end] + end_idx

    return new_start, new_end  # new range of IDR (inclusive)

def mask_seq(seq, new_start_idx, new_end_idx, i, num_unpadded_rows):
    if i < num_unpadded_rows:
        idr_range = new_end_idx - new_start_idx
        masked_seq = seq[0:new_start_idx] + '#' * idr_range + seq[new_end_idx:]
    else:
        masked_seq = seq
    return masked_seq


if __name__ == '__main__':
    main()
import argparse
import evodiff
import os
import numpy as np
from tqdm import tqdm
import pathlib
import glob
from evodiff.data import A3MMSADataset, IDRDataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch
from sequence_models.collaters import MSAAbsorbingCollater
from evodiff.collaters import D3PMCollaterMSA
from sequence_models.constants import MSA_ALPHABET
from evodiff.utils import Tokenizer
home = str(pathlib.Path.home())

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('config_fpath')
    #parser.add_argument('out_fpath', type=str, nargs='?',
    #                    default=os.getenv('AMLT_OUTPUT_DIR', '/tmp') + '/')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-off', '--offset', default=0, type=int,
                        help='Number of GPU devices to skip.')
    parser.add_argument('--model-type', type=str, default='msa_oa_dm_maxsub')
    parser.add_argument('--dataset', type=str, default='openfold')
    parser.add_argument('--batch-size', type=int, default=1) # batch-size (on amlt use 1)
    parser.add_argument('--n-sequences', type=int, default=64)
    parser.add_argument('--seq-length', type=int, default=512)
    parser.add_argument('--penalty-value', type=float, default=0) # Default no penalty /=1 on gap generation
    parser.add_argument('--run', type=int, default=0) # Default no penalty /=1 on gap generation
    parser.add_argument('--subsampling', type=str, default='MaxHamming')
    parser.add_argument('--delete-prev', action='store_true')  # Will delete previous generated sequences that start with generated* in main folder
    parser.add_argument('--start-query', action='store_true') # if starting from query -> gen msa
    parser.add_argument('--start-msa', action='store_true') # if starting from msa -> gen query
    parser.add_argument('--amlt', action='store_true') # if running on amlt
    args = parser.parse_args()

    #_ = torch.manual_seed(0)
    np.random.seed(0)

    torch.cuda.set_device(args.gpus + args.offset)
    device = torch.device('cuda:' + str(args.gpus + args.offset))

    d3pm = False
    if args.model_type == 'msa_oa_dm_randsub':
        checkpoint = evodiff.pretrained.MSA_OA_DM_RANDSUB()
        #selection_type = 'random'
        mask_id = checkpoint[2].mask_id
        pad_id = checkpoint[2].pad_id
    elif args.model_type == 'msa_oa_dm_maxsub':
        checkpoint = evodiff.pretrained.MSA_OA_DM_MAXSUB()
        #selection_type = 'MaxHamming'
        mask_id = checkpoint[2].mask_id
        pad_id = checkpoint[2].pad_id
    elif args.model_type == 'esm_msa_1b':
        checkpoint = evodiff.pretrained.ESM_MSA_1b()
        #selection_type = 'MaxHamming'
        mask_id = checkpoint[2].mask_idx
        pad_id = checkpoint[2].padding_idx
    elif args.model_type == 'msa_d3pm_blosum_maxsub':
        checkpoint = evodiff.pretrained.MSA_D3PM_BLOSUM_MAXSUB()
        d3pm=True
        mask_id = checkpoint[2].mask_id
        pad_id = checkpoint[2].pad_id
    elif args.model_type == 'msa_d3pm_blosum_randsub':
        checkpoint = evodiff.pretrained.MSA_D3PM_BLOSUM_RANDSUB()
        d3pm = True
        mask_id = checkpoint[2].mask_id
        pad_id = checkpoint[2].pad_id
    elif args.model_type == 'msa_d3pm_uniform_maxsub':
        checkpoint = evodiff.pretrained.MSA_D3PM_UNIFORM_MAXSUB()
        d3pm = True
        mask_id = checkpoint[2].mask_id
        pad_id = checkpoint[2].pad_id
    elif args.model_type == 'msa_d3pm_uniform_randsub':
        checkpoint = evodiff.pretrained.MSA_D3PM_UNIFORM_RANDSUB()
        d3pm=True
        mask_id = checkpoint[2].mask_id
        pad_id = checkpoint[2].pad_id
    else:
        raise Exception("Please select either msa_or_ar_randsub, msa_oa_oar_maxsub, msa_d3pm_blosum_maxsub, "
                        "msa_d3pm_blosum_randsub, msa_d3pm_uniform_maxsub, msa_d3pm_uniform_randsub,"
                        "or esm_msa_1b baseline. You selected:", args.model_type)

    try:
        data_top_dir = os.getenv('AMLT_DATA_DIR') + '/data/data/data/' # TODO i messed up my amulet storage - this works for now
        data_dir = data_top_dir
        data_dir += args.dataset + '/'
        ptjob = True
    except:
        data_top_dir = 'data/'
        #print(data_top_dir)
        data_dir = data_top_dir
        data_dir += args.dataset + '/'
        ptjob = False

    if d3pm:
        model, collater, tokenizer, scheme, timestep, Q_bar, Q = checkpoint
        Q_bar = Q_bar.to(device)
        Q = Q.to(device)
    else:
        model, collater, tokenizer, scheme = checkpoint

    model = model.eval().to(device)

    #project_dir = home + '/Desktop/DMs/'

    if args.start_query and args.start_msa:
        raise Exception("Can only choose either start-query or start-msa NOT both, to generate from scratch omit flags")

    if args.amlt:
        home = os.getenv('AMLT_OUTPUT_DIR', '/tmp') + '/'
        out_fpath = home
    else:
        home = str(pathlib.Path.home()) + '/Desktop/DMs/'
        top_dir = home
        out_fpath = home + args.model_type + '/gen-'+str(args.run) + '/'

    if not os.path.exists(out_fpath):
        os.makedirs(out_fpath)

    if args.delete_prev:
        filelist = glob.glob(out_fpath+'generated*')
        filelist += glob.glob(out_fpath+'msas/*generated*')
        filelist += glob.glob(out_fpath+'valid*')
        for file in filelist:
            os.remove(file)
            print("Deleting", file)
    if args.penalty_value > 0:
        print("Penalizing GAPS by factor of", 1+args.penalty_value)
    else:
        print("Not penalizing GAPS")

    if scheme == 'mask':
        sample, _string = generate_msa(model, tokenizer, args.batch_size, args.n_sequences, args.seq_length,
                                      penalty_value=args.penalty_value, device=device, start_query=args.start_query,
                                       start_msa=args.start_msa,
                                      data_top_dir=data_top_dir, selection_type=args.subsampling, out_path=out_fpath)
    elif scheme == 'd3pm':
        sample, _string = generate_msa_d3pm(model, args.batch_size, args.n_sequences, args.seq_length,
                                           Q_bar=Q_bar, Q=Q, tokenizer=Tokenizer(), data_top_dir=data_top_dir,
                                           selection_type=args.subsampling, out_path=out_fpath,
                                           max_timesteps=timestep, start_query=args.start_query,
                                           no_step=False, penalty_value=args.penalty_value, device=device)


    for count, msa in enumerate(_string):
        fasta_string = ""
        with open(out_fpath + 'generated_msas.a3m', 'a') as f:
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
        np.save(out_fpath+'generated_msas', np.array(sample.cpu()))


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
            seq_len = len(query_sequences[i])
            print("PAD ID", tokenizer.pad_id)
            src[i][0][:seq_len] = query_sequences[i]
            padding = torch.full((n_sequences, seq_length-seq_len), fill_value=tokenizer.pad_id)
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
            p_softmax /= penalty
            p_sample = torch.multinomial(input=p_softmax, num_samples=1)
            p_sample = p_sample.squeeze()
            sample[:, random_x, random_y] = p_sample
    untokenized = [[tokenizer.untokenize(msa.flatten())] for msa in sample]
    return sample, untokenized # return output and untokenized output

def generate_query_oadm_msa_simple(path_to_msa, model, tokenizer, n_sequences, seq_length, batch_size=1, penalty_value=2, device='gpu',
                 start_msa=True, selection_type='MaxHamming'):
    mask_id = tokenizer.mask_id
    src = torch.full((batch_size, n_sequences, seq_length), fill_value=mask_id)

    valid_msas = []
    query_sequences = []
    for i in range(batch_size):
        #print(path_to_msa)
        valid_msa, query_sequence = evodiff.data.subsample_msa(path_to_msa, n_sequences=n_sequences,
                                                               max_seq_len=seq_length, selection_type=selection_type)
        valid_msa = torch.tensor(np.array([tokenizer.tokenizeMSA(seq) for seq in valid_msa]))
        valid_msas.append(valid_msa)
        query_sequences.append(query_sequence)

    for i in range(batch_size):
        seq_len = len(query_sequences[i])
        src[i, 1:n_sequences, :seq_len] = valid_msas[i][1:n_sequences, :seq_len].squeeze()
        padding = torch.full((n_sequences, seq_length-seq_len), fill_value=tokenizer.pad_id)
        src[i, :, seq_len:] = padding
        x_indices = np.arange(0,1)
        y_indices = np.arange(seq_len)
    src = src.to(device)
    sample = src.clone()
    if start_msa:
        all_ind = np.transpose([np.tile(x_indices, len(y_indices)), np.repeat(y_indices, len(x_indices))])
    np.random.shuffle(all_ind)

    # ONLY USING ON BATCH_SIZE=1 for now
    with torch.no_grad():
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
            p_softmax /= penalty
            p_sample = torch.multinomial(input=p_softmax, num_samples=1)
            p_sample = p_sample.squeeze()
            sample[:, random_x, random_y] = p_sample
    untokenized = [[tokenizer.untokenize(msa[0])] for msa in sample] # return query sequence only
    return sample, untokenized # return query sequences only

def generate_msa_d3pm(model, batch_size, n_sequences, seq_length, Q_bar=None, Q=None, tokenizer=Tokenizer(),
                      start_query=False, data_top_dir='../data', selection_type='MaxHamming', out_path='../ref/',
                      max_timesteps=500, no_step=False, penalty_value=0, device='gpu'):
    sample = torch.randint(0, tokenizer.K, (batch_size, n_sequences, seq_length))
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
                A = torch.mm(x_t_b, torch.t(Q[t]))  # [P x K]
                Q_expand = Q_bar[t-1].unsqueeze(0).expand(A.shape[0], tokenizer.K, tokenizer.K)  # [ P x K x K]
                B_pred = torch.mul(p_current.unsqueeze(2), Q_expand)
                q_t = torch.mul(A.unsqueeze(1), B_pred)  # [ P x K x K ]
                p_theta_marg = torch.bmm(torch.transpose(q_t, 1,2),  p_current.unsqueeze(2)).squeeze()  # this marginalizes over dim=2
                p_theta_marg = p_theta_marg / p_theta_marg.sum(axis=1, keepdim=True)
                penalty = torch.ones(p_theta_marg.shape).to(p_theta_marg.device)
                penalty[:, -1] += penalty_value
                p_theta_marg /= penalty
                x_tminus1_temp = torch.multinomial(p_theta_marg[:, :], num_samples=1).squeeze()
                x_tminus1_temp[:seq_length] = torch.multinomial(p_theta_marg[:seq_length,:-1], num_samples=1).squeeze() # NO GAPS in query
                if start_query:
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
    return sample, untokenized


def get_valid_data(data_top_dir, num_seqs, arg_mask, data_dir='openfold/', selection_type='MaxHamming', n_sequences=64, max_seq_len=512,
                   out_path='../DMs/ref/'):
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

    return valid_msas, query_msas, tokenizer


if __name__ == '__main__':
    main()
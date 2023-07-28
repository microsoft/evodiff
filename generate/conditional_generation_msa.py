from evodiff.pretrained import MSA_OA_AR_MAXSUB, MSA_OA_AR_RANDSUB, ESM_MSA_1b
import numpy as np
import argparse
import urllib.request
import torch
import os
import esm.inverse_folding
from evodiff.utils import Tokenizer, run_omegafold, clean_pdb, run_tmscore
import pathlib
from sequence_models.utils import parse_fasta
from tqdm import tqdm
import pandas as pd
import random
from evodiff.plot import aa_reconstruction_parity_plot
from scipy.spatial.distance import hamming, cdist

def main():
    # set seeds
    _ = torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='D3PM_BLOSUM_38M',
                        help='Choice of: msa_oa_ar_randsub, msa_oa_ar_maxsub, esm_msa_1b')
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--cond-task', type=str, default='scaffold',
                        help="Choice of 'scaffold' or 'idr'")
    parser.add_argument('--pdb', type=str, default=None,
                        help="If using cond-task=scaffold, provide a PDB code and motif indexes")
    parser.add_argument('--start-idxs', type=int, action='append',
                        help="If using cond-task=scaffold, provide start and end indexes for motif being scaffolded\
                             If defining multiple motifs, supply the start and end -idx motif as a new argument\
                              ex: --start-idx 3 --end-idx 10 --start-idx 20 --end-idx 25\
                              indexes are inclusive of both start and end values.\
                              WARNING: PDBs are OFTEN indexed at a number that is not 0. If your PDB file begins at 4\
                              and the motif you want to query is residues 5 to 10, as defined by the PDB, your inputs to\
                              this code should be --start-idx 1 and --end-idx 6")
    parser.add_argument('--end-idxs', type=int, action='append')
    parser.add_argument('--num-seqs', type=int, default=10,
                        help="Number of sequences generated per scaffold length")
    parser.add_argument('--max-seq-len', type=int, default=150,
                        help="Max seq len to splice from MSA")
    parser.add_argument('--n-sequences', type=int, default=64,
                        help="Number of seqs to subsample from MSA")
    parser.add_argument('--random-baseline', action='store_true')
    parser.add_argument('--query-only', action='store_true')
    parser.add_argument('--amlt', action='store_true')
    parser.add_argument('--single-res-domain', action='store_true', help="if start-idx = end-idx make sure to use single-res-domain flag or else you will get errors")
    args = parser.parse_args()

    args.start_idxs.sort()
    args.end_idxs.sort()

    if args.random_baseline:
        args.model_type = 'msa_oa_ar_randsub' # placeholder

    if args.model_type == 'msa_oa_ar_randsub':
        checkpoint = MSA_OA_AR_RANDSUB()
        selection_type = 'random'
        mask_id = checkpoint[2].mask_id
        pad_id = checkpoint[2].pad_id
    elif args.model_type == 'msa_oa_ar_maxsub':
        checkpoint = MSA_OA_AR_MAXSUB()
        selection_type = 'MaxHamming'
        mask_id = checkpoint[2].mask_id
        pad_id = checkpoint[2].pad_id
    elif args.model_type == 'esm_msa_1b':
        checkpoint = ESM_MSA_1b()
        selection_type = 'MaxHamming'
        mask_id = checkpoint[2].mask_idx
        pad_id = checkpoint[2].padding_idx
    else:
        raise Exception("Please select either msa_or_ar_randsub, msa_oa_oar_maxsub, or esm_msa_1b baseline. You selected:", args.model_type)

    model, collater, tokenizer, scheme = checkpoint
    model.eval().cuda()

    torch.cuda.set_device(args.gpus)
    device = torch.device('cuda:' + str(args.gpus))

    if args.amlt:
        home = os.getenv('AMLT_OUTPUT_DIR', '/tmp') + '/'
        top_dir = ''
        out_fpath = home
    else:
        home = str(pathlib.Path.home()) + '/Desktop/DMs/'
        top_dir = home

        if not args.random_baseline:
            out_fpath = home + args.model_type + '/' + args.pdb +'/'
        else:
            out_fpath = home + 'random-baseline/' + args.pdb +'/'
        if not os.path.exists(out_fpath):
            os.makedirs(out_fpath)

    data_top_dir = top_dir + 'data/'

    # After cond gen, run omegafold  # TODO for debug only
    print("Finished generation, starting omegafold")
    run_omegafold(out_fpath, fasta_file="generated_samples_string.fasta")

    if args.cond_task == 'idr':
        #TODO Finish IDR
        sample, string, queries, sequences = generate_idr_msa(model, data_top_dir, tokenizer=tokenizer,
                                                          penalty=args.penalty, batch_size=1, device=device)
    elif args.cond_task == 'scaffold':
        strings = []
        start_idxs = []
        end_idxs = []
        scaffold_lengths = []
        for i in range(args.num_seqs): # no batching
            string, new_start_idx, new_end_idx, seq_len = generate_scaffold_msa(args.model_type, model, args.pdb,
                                                                                args.start_idxs, args.end_idxs,
                                                                       data_top_dir, tokenizer, device=device,
                                                                       random_baseline=args.random_baseline,
                                                                       query_only=args.query_only,
                                                                       max_seq_len=args.max_seq_len,
                                                                       n_sequences=args.n_sequences,
                                                                       selection_type=selection_type,
                                                                       mask=mask_id, pad=pad_id)
            #print("STRING", string)
            strings.append(string)
            start_idxs.append(new_start_idx)
            end_idxs.append(new_end_idx)
            scaffold_lengths.append(seq_len)


    save_df = pd.DataFrame(list(zip(strings, start_idxs, end_idxs, scaffold_lengths)), columns=['seqs', 'start_idxs', 'end_idxs', 'scaffold_lengths'])
    save_df.to_csv(out_fpath+'motif_df.csv', index=True)

    with open(out_fpath + 'generated_samples_string.csv', 'w') as f:
        for _s in strings:
            f.write(_s[0]+"\n")
    with open(out_fpath + 'generated_samples_string.fasta', 'w') as f:
        for i, _s in enumerate(strings):
            f.write(">SEQUENCE_" + str(i) + "\n" + str(_s[0]) + "\n")

    # After cond gen, run omegafold
    print("Finished generation, starting omegafold")
    run_omegafold(out_fpath, fasta_file="generated_samples_string.fasta")

    print("Cleaning PDBs")
    # clean PDB for TMScore analysis
    clean_pdb(os.path.join(out_fpath, 'pdb/'), data_top_dir, args.pdb)

    print("Getting TM scores")
    # Get TMscores
    run_tmscore(out_fpath, args.pdb, args.num_seqs, path_to_tmscore=top_dir+'TMscore', amlt=args.amlt)


def get_MSA(filename, tokenizer):
    parsed_msa, msa_names = parse_fasta(filename, return_names=True)

    # Clean MSA
    aligned_msa = [[char for char in seq if (char.isupper() or char == '-') and not char == '.'] for seq in parsed_msa]
    aligned_msa = [''.join(seq) for seq in aligned_msa]

    tokenized_msa = [tokenizer.tokenizeMSA(seq) for seq in aligned_msa]
    tokenized_msa = np.array([l.tolist() for l in tokenized_msa])
    return tokenized_msa

def subsample_MSA(tokenized_msa, start_idxs, end_idxs, tokenizer, query_idx=0, max_seq_len=512, n_sequences=64, selection_type='random'):
    """
    Inputs
    tokenized_msa: tokenized MSA
    start_idx: motif start idxs
    end_idx: motif end idxs
    query_idx: query sequence index (default=0)
    tokenizer: tokenizer corresponding to loaded checkpoint, and tokenized MSA
    max_seq_len: maximum length of MSA to subsample (int)
    n_sequences: maximum sequences to subsample (int)
    selection_type: either 'MaxHamming' or 'random' subsampling scheme for MSAs

    Outputs
    output: untokenized msas (list)
    sliced_start_idx: new IDR start index of MSA
    sliced_end_idx: new IDR end index of MSA
    msa_n_sequences: number of sequences in msa (will be less than or = n_sequences)
    """

    msa_seq_len = len(tokenized_msa[0])
    motif_start = start_idxs[0]
    motif_end = end_idxs[-1]
    original_motif = [tokenizer.untokenize(tokenized_msa[0][start_idxs[i]:end_idxs[i]]) for i in range(len(start_idxs))]

    # Slice around motif
    if msa_seq_len > max_seq_len:
        # If seq len larger than max, center motif in slice
        motif_len = motif_end - motif_start
        buffer = int((max_seq_len - motif_len)/2)
        if motif_start - buffer < 0: # if MOTIF at beginning of seq
            print("BEGINNING")
            slice_start = 0
            slice_end = max_seq_len
            sliced_start_idxs = start_idxs
            sliced_end_idxs = end_idxs
        elif motif_end + buffer > msa_seq_len: # if MOTIF at end of seq
            print("END")
            slice_start = msa_seq_len - max_seq_len
            slice_end = msa_seq_len
            sliced_end_idxs = [end_idx - slice_start for end_idx in end_idxs]
            sliced_start_idxs = [start_idx - slice_start for start_idx in start_idxs]
        else: # center IDR
            print("CENTER")
            slice_start = motif_start - buffer
            slice_end = motif_end + buffer
            sliced_start_idxs = [start_idx - slice_start for start_idx in start_idxs]
            sliced_end_idxs = [end_idx - slice_start for end_idx in end_idxs]
        # print("SLICING INDEX", slice_start, slice_end)
        # print("OLD INDEX", start_idx, end_idx)
        print("NEW INDEX, adjust slice", sliced_start_idxs, sliced_end_idxs)
        #seq_len = max_seq_len
    else:
        slice_start = 0
        slice_end = msa_seq_len
        sliced_start_idxs = start_idxs
        sliced_end_idxs = end_idxs

    # Slice to model constraints
    sliced_msa_seq = tokenized_msa[:, slice_start: slice_end]
    # Remove query from array
    sliced_msa_seq = np.append(sliced_msa_seq[:query_idx], sliced_msa_seq[query_idx+1:], axis=0)
    # Query Sequence
    anchor_seq = tokenized_msa[query_idx, slice_start:slice_end]  # This is the query sequence
    sliced_msa = [seq for seq in sliced_msa_seq if (list(set(seq)) != [tokenizer.gap_id])]
    msa_num_seqs = len(sliced_msa) + 1 # +1 accounts for query

    if msa_num_seqs > n_sequences:
        #msa_n_sequences = n_sequences
        if selection_type == 'random':
            print("Using random subsampling")
            random_idx = np.random.choice(msa_num_seqs-1, size=n_sequences-1, replace=False)
            anchor_seq = np.expand_dims(anchor_seq, axis=0)
            output = np.concatenate((anchor_seq, np.array(sliced_msa)[random_idx.astype(int)]), axis=0)
        elif selection_type == "MaxHamming":
            print("using MaxHamming subsampling")
            output = [list(anchor_seq)]
            msa_subset = sliced_msa
            msa_ind = np.arange(msa_num_seqs-1)
            random_ind = np.random.choice(msa_ind)
            random_seq = sliced_msa[random_ind]
            output.append(list(random_seq))
            random_seq = np.expand_dims(random_seq, axis=0)
            msa_subset = np.delete(msa_subset, (random_ind), axis=0)
            m = len(msa_ind) - 1
            distance_matrix = np.ones((n_sequences - 2, m))
            for i in range(n_sequences - 2):
                curr_dist = cdist(random_seq, msa_subset, metric='hamming')
                curr_dist = np.expand_dims(np.array(curr_dist), axis=0)  # shape is now (1,msa_num_seqs)
                #print(curr_dist.shape)
                distance_matrix[i] = curr_dist
                col_min = np.min(distance_matrix, axis=0)  # (1,num_choices)
                max_ind = np.argmax(col_min)
                random_ind = max_ind
                random_seq = msa_subset[random_ind]
                output.append(list(random_seq))
                random_seq = np.expand_dims(random_seq, axis=0)
                msa_subset = np.delete(msa_subset, random_ind, axis=0)
                distance_matrix = np.delete(distance_matrix, random_ind, axis=1)
    else:
        #msa_n_sequences = msa_num_seqs
        output = np.full(shape=(n_sequences, max_seq_len), fill_value=tokenizer.gap_id) # Treat short seqs as being algined with large gaps
        output[0:1, :len(anchor_seq)] = anchor_seq
        output[1:msa_num_seqs, :len(anchor_seq)] = sliced_msa
        #output = np.concatenate((np.array(anchor_seq).reshape(1,-1), np.array(sliced_msa)), axis=0)

    output_motif = [tokenizer.untokenize(output[0][sliced_start_idxs[i]:sliced_end_idxs[i]]) for i in range(len(sliced_start_idxs))]
    print("og", original_motif)
    print("out", output_motif)
    assert original_motif == output_motif, "RE-SLICED MOTIFS DON'T MATCH, CHECK INDEXING"
    output = [tokenizer.untokenize(seq) for seq in output]
    # print(len(output), len(output[0]))
    return output, sliced_start_idxs, sliced_end_idxs, original_motif

def get_masked_locations(query_sequence, sliced_start_idxs, sliced_end_idxs, pad_id):
    "Return list of masked indices given a list of starting and ending indeces for motifs"
    #input_mask = (query_sequence != pad_id)
    seq_len = len(query_sequence)
    all_index = np.arange(seq_len)
    list_motif = [list(range(sliced_start_idxs[i], sliced_end_idxs[i])) for i in range(len(sliced_start_idxs))]
    list_motif = [item for sublist in list_motif for item in sublist]
    list_masked = [x for x in all_index if x not in list_motif]
    print(list_masked)
    return list_masked

def mask_sequence(seq, mask_locations, mask_id):
    masked_seq = []
    for i in range(len(seq)):
        if i in mask_locations:
            masked_seq.append(mask_id)
        else:
            masked_seq.append(seq[i])
    return masked_seq

def tokenize_msa(model_type, untokenized, tokenizer):
    if model_type == 'msa_oa_ar_maxsub' or model_type == 'msa_oa_ar_randsub':
        return [tokenizer.tokenizeMSA(seq) for seq in untokenized]
    elif model_type == 'esm_msa_1b':
        src = []
        for i, seq in enumerate(untokenized):
            new_seq = [tokenizer.cls_idx] + [tokenizer.get_idx(c) for c in [*seq]] + [tokenizer.eos_idx]
            src.append(new_seq)
        return src

def untokenize_msa(model_type, tokenized, tokenizer):
    if model_type == 'msa_oa_ar_maxsub' or model_type == 'msa_oa_ar_randsub':
        return tokenizer.untokenize(tokenized)
    elif model_type == 'esm_msa_1b':
        return ''.join([tokenizer.get_tok(s) for s in tokenized[1:-1]])


def generate_scaffold_msa(model_type, model, PDB_ID, motif_start_idxs, motif_end_idxs, data_top_dir, tokenizer, query_only=True,
                      device='gpu', random_baseline=False,  max_seq_len=512, n_sequences=64, selection_type='random',
                      mask=0, pad=1):
    motif_end_idxs = [i + 1 for i in motif_end_idxs]  # inclusive of final residue
    if random_baseline:
        train_prob_dist = aa_reconstruction_parity_plot(data_top_dir+'../', 'reference/', 'placeholder.csv', gen_file=False)

    tokenized_msa = get_MSA(data_top_dir + '/scaffolding-msas/' + PDB_ID+'.a3m', tokenizer=Tokenizer())
    sliced_msa, sliced_start_idxs, sliced_end_idxs, original_motif = subsample_MSA(tokenized_msa, motif_start_idxs, motif_end_idxs,
                                                                 Tokenizer(), query_idx=0, max_seq_len=max_seq_len,
                                                                 n_sequences=n_sequences, selection_type=selection_type)

    print("INPUT MSA", sliced_msa[0])

    # Now tokenize using tokenizer of choice
    sliced_msa = tokenize_msa(model_type, sliced_msa, tokenizer)
    query_sequence = sliced_msa[0]  # ensure query is first seq -> not true for IDRs

    if model_type == 'esm_msa_1b':
        seq_len = len(query_sequence)-2
        mask_locations = get_masked_locations(query_sequence[1:-1], sliced_start_idxs, sliced_end_idxs, pad_id=pad)
        mask_locations = [i + 1 for i in mask_locations]
        max_token = len(tokenizer)
    else:
        seq_len = len(query_sequence)
        mask_locations = get_masked_locations(query_sequence, sliced_start_idxs, sliced_end_idxs, pad_id=pad)
        max_token = tokenizer.K - 1

    masked_loc_y = mask_locations
    # Mask out non-motif residues in query sequence of msa
    sliced_msa[0] = mask_sequence(query_sequence, mask_locations, mask)
    masked_loc_x = [0]
    query_ind = np.transpose([np.tile(masked_loc_x, len(masked_loc_y)), np.repeat(masked_loc_y, len(masked_loc_x))])
    np.random.shuffle(query_ind)
    if not query_only:
        # Mask out non-motif residues in query sequence of msa
        sliced_msa = [mask_sequence(seq, mask_locations, mask) for seq in sliced_msa]
        masked_loc_x = np.arange(1, n_sequences)  # len of MSA ; num sequences
        all_ind = np.transpose([np.tile(masked_loc_x, len(masked_loc_y)), np.repeat(masked_loc_y, len(masked_loc_x))])
        np.random.shuffle(all_ind)

    sample = torch.tensor(sliced_msa).unsqueeze(0)
    sample = sample.to(device)
    with torch.no_grad():
        if not query_only:
            # First gen MSA
            for i in tqdm(all_ind):
                random_x, random_y = i
                if model_type == 'esm_msa_1b':
                    results = model(sample, repr_layers=[33], return_contacts=True)
                    preds = results["logits"]
                else:
                    preds = model(sample)  # Output shape of preds is (BS=1, N=64, L, n_tokens=31)
                p = preds[:, random_x, random_y, :]  # for first row don't let p_softmax predict gaps
                p_softmax = torch.nn.functional.softmax(p, dim=1)
                p_sample = torch.multinomial(input=p_softmax, num_samples=1)
                p_sample = p_sample.squeeze()
                sample[:, random_x, random_y] = p_sample
                #print(untokenize_msa(model_type, sample[0][0], tokenizer))
        # Then gen query seq
        for i in tqdm(query_ind):
            random_x, random_y = i
            #print(random_x, random_y, len(sample[0][0]))
            if model_type == 'esm_msa_1b':
                results = model(sample, repr_layers=[33], return_contacts=True)
                preds = results["logits"]
            else:
                preds = model(sample)  # Output shape of preds is (BS=1, N=64, L, n_tokens=31)
            p = preds[:, random_x, random_y, :max_token] # for first row don't let p_softmax predict gaps
            p_softmax = torch.nn.functional.softmax(p, dim=1)
            p_sample = torch.multinomial(input=p_softmax, num_samples=1)
            p_sample = p_sample.squeeze()
            sample[:, random_x, random_y] = p_sample
            print(untokenize_msa(model_type, sample[0][0], tokenizer))

    untokenized = [untokenize_msa(model_type, sample[0][0], tokenizer)] # only return query sequence
    print(untokenized)

    return untokenized, sliced_start_idxs, [i - 1 for i in sliced_end_idxs], seq_len  # return output and untokenized output, re-indexed motif starts and ends (ends-1 for rmsd analyis)


def generate_idr_msa(model, tokenizer, n_sequences, seq_length, penalty_value=2, device='gpu', index=0,
                 start_query=False, data_top_dir='../data', selection_type='MaxHamming', out_path='../ref/'):
    src, start_idx, end_idx, original_idr, num_sequences = get_IDR_MSAs(index, data_top_dir, tokenizer, max_seq_len=seq_length,
                                                         n_sequences=n_sequences, out_path=out_path,
                                                         selection_type=selection_type)
    src = torch.tensor(src).unsqueeze(0) # Make batchsize 1

    masked_loc_x = np.arange(num_sequences) # len of MSA ; num sequences
    masked_loc_y = np.arange(start_idx, end_idx)
    all_ind = np.transpose([np.tile(masked_loc_x, len(masked_loc_y)), np.repeat(masked_loc_y, len(masked_loc_x))])
    np.random.shuffle(all_ind)

    sample = src.clone()
    sample = sample.to(device)

    with torch.no_grad():
        for i in tqdm(all_ind):
            #print(i)
            random_x, random_y = i
            print(sample.shape)
            preds = model(sample)  # Output shape of preds is (BS=1, N=64, L, n_tokens=31)
            print("preds", preds.shape)
            print(random_x, random_y)
            p = preds[:, random_x, random_y, :]
            # if random_x == 0:  # for first row don't let p_softmax predict gaps
            #     p = preds[:, random_x, random_y, :tokenizer.K - 1]
            p_softmax = torch.nn.functional.softmax(p, dim=1)
            # Penalize gaps
            penalty = torch.ones(p.shape).to(p.device)
            penalty[:, -1] += penalty_value
            # print(p_softmax)
            p_softmax /= penalty
            # print(p_softmax)
            p_sample = torch.multinomial(input=p_softmax, num_samples=1)
            p_sample = p_sample.squeeze()
            sample[:, random_x, random_y] = p_sample
            print(tokenizer.untokenize(sample[0][0]))
    print(sample.shape)
    #print([tokenizer.untokenize(seq) for seq in sample[0]])
    new_idr = [tokenizer.untokenize(seq[start_idx:end_idx]) for seq in sample[0]]
    untokenized = [[tokenizer.untokenize(msa.flatten())] for msa in sample[0]]

    #print(untokenized[0])
    return sample, untokenized, original_idr, new_idr  # return output and untokenized output

def mask_seq(seq, new_start_idx, new_end_idx, i, num_unpadded_rows):
    if i < num_unpadded_rows:
        idr_range = new_end_idx - new_start_idx
        masked_seq = seq[0:new_start_idx] + '#' * idr_range + seq[new_end_idx:]
    else:
        masked_seq = seq
    return masked_seq

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

def preprocess_IDR_data(data_top_dir):
    data_dir = data_top_dir + 'human_idr_alignments/'
    all_files = os.listdir(data_dir + 'human_protein_alignments')
    index_file = pd.read_csv(data_dir + 'human_idr_boundaries.tsv', delimiter='\t')

    # Filter out IDRs > 250 residues in length
    index_file['LENGTHS'] = list(index_file['END'] - index_file['START'])
    print("BEFORE FILTERING OUT LONG IDRS", len(index_file))
    index_file = index_file[index_file['LENGTHS'] <= 250]
    print("AFTER FILTERING LONG IDRS", len(index_file))
    # print(index_file.head())
    # import pdb; pdb.set_trace()
    # print(len(index_file), "TOTAL IDRS")
    # REFILTER FOR GAPPED MSAs
    #index_file = index_file[:3] # TODO delete after debug
    new_starts = []
    new_ends = []
    for index, row in index_file.iterrows():
        msa_file = [file for i, file in enumerate(all_files) if row['OMA_ID'] in file][0]
        msa_data, msa_names = parse_fasta(data_dir + 'human_protein_alignments/' + msa_file, return_names=True)
        query_idx = [i for i, name in enumerate(msa_names) if name == row['OMA_ID']][0]  # get query index
        seq_only = msa_data[query_idx].replace("-", "")
        start_idx = row['START']
        end_idx = row['END']
        new_start_idx, new_end_idx = reindex_IDR(start_idx, end_idx, seq_only, msa_data[query_idx])
        new_start_idx -= 1  # original range starts at 1, inclusive
        new_starts.append(new_start_idx)
        new_ends.append(new_end_idx)
        # # VERIFY REINDEXED IDR IS CORRECT
        # print(row['IDR_SEQ'])
        # print(msa_data[query_idx][new_start_idx:new_end_idx])
        # print(msa_data[query_idx][new_start_idx:new_end_idx].replace('-', ''))
        #import pdb; pdb.set_trace()
    #print(len(new_starts), len(new_ends), len(index_file))
    index_file['GAP START'] = new_starts
    index_file['GAP END'] = new_ends
    index_file['GAP LENGTHS'] = list(index_file['GAP END'] - index_file['GAP START'])
    index_file = index_file[index_file['GAP LENGTHS'] <= 250]
    print("AFTER FILTERING LONG GAP IDRS", len(index_file))
    index_file.to_csv(data_dir + 'human_idr_boundaries_gap.tsv', sep='\t')

def get_IDR_MSAs(index, data_top_dir, tokenizer, max_seq_len=512, n_sequences=64, out_path='', selection_type='random'):
    # GET IDRS
    data_dir = data_top_dir + 'human_idr_alignments/'
    all_files = os.listdir(data_dir + 'human_protein_alignments')
    if not os.path.exists(data_dir + 'human_idr_boundaries_gap.tsv'):
        preprocess_IDR_data(data_top_dir)
    index_file = pd.read_csv(data_dir + 'human_idr_boundaries_gap.tsv', delimiter='\t')

    row = index_file.iloc[index]
    # Get MSA
    msa_file = [file for i, file in enumerate(all_files) if row['OMA_ID'] in file][0]
    msa_data, new_start_idx, new_end_idx, num_sequences = subsample_IDR_MSA(row, msa_file, data_dir, tokenizer, max_seq_len=max_seq_len, n_sequences=n_sequences,
                                 selection_type=selection_type)
    # new_start_idx = row['GAP START']
    # new_end_idx = row['GAP END']
    #print("new index?", new_start_idx, new_end_idx)
    # print("ENTIRE QUERY", msa_data[0])
    # print("ENTIRE IDR", msa_data[0][new_start_idx:new_end_idx])
    # print("PRE MASK IDR", msa_data[0][new_start_idx:new_end_idx].replace("-",""))
    # MASK out IDR
    masked_msa = [mask_seq(seq, new_start_idx, new_end_idx, i, num_sequences) for i, seq in enumerate(msa_data)]
    #print(len(masked_msa))
    #print(masked_msa)
    #import pdb; pdb.set_trace()
    #print("ENTIRE MASKED QUERY", masked_msa[0])
    # import pdb; pdb.set_trace()
    original_msa_idr = msa_data
    tokenized_msa = [tokenizer.tokenizeMSA(seq) for seq in masked_msa]
    tokenized_msa = np.array([l.tolist() for l in tokenized_msa])

    print(row)
    #print("true IDR", row['IDR_SEQ'])

    with open(out_path + 'valid_msas.a3m', 'a') as f:
        for i, msa in enumerate(original_msa_idr):
            #print(i, msa)
            if i == 0 :
                f.write(">SEQUENCE_" + str(i) + "\n" + str(msa) + "\n")
            else:
                f.write(">tr \n" + str(msa) + "\n" )
        f.close()
    with open(out_path + 'valid_idr.a3m', 'a') as f:
        for i, msa in enumerate(original_msa_idr):
            if i == 0 :
                print("CAPTURED IDR", msa[new_start_idx:new_end_idx])
                f.write(">SEQUENCE_" + str(i) + "\n" + str(msa[new_start_idx:new_end_idx]) + "\n")
            else:
                f.write(">tr \n" + str(msa[new_start_idx:new_end_idx]) + "\n" )
        f.close()

    return tokenized_msa, new_start_idx, new_end_idx, original_msa_idr, num_sequences

def subsample_IDR_MSA(row, filename, data_dir, tokenizer, max_seq_len=512, n_sequences=64, selection_type='random'):
    ## TODO CAN use a general subsample MSA here -> try to recode
    """
    Inputs
    row:
    filename: name of MSA file (str)
    data_dir: directory where data is located (str)
    tokenizer: tokenizer corresponding to loaded checkpoint
    max_seq_len: maximum length of MSA to subsample (int)
    n_sequences: maximum sequences to subsample (int)
    selection_type: either 'MaxHamming' or 'random' subsampling scheme for MSAs

    Outputs
    output: untokenized msas (list)
    sliced_idr_start_idx: new IDR start index of MSA
    sliced_idr_end_idx: new IDR end index of MSA
    msa_n_sequences: number of sequences in msa (will be less than or = n_sequences)
    """
    parsed_msa, msa_names = parse_fasta(data_dir + 'human_protein_alignments/' + filename, return_names=True)
    # Get query
    query_idx = [i for i, name in enumerate(msa_names) if name == row['OMA_ID']][0]  # get query index

    new_start_idx = row['GAP START']
    new_end_idx = row['GAP END']

    aligned_msa = [[char for char in seq if (char.isupper() or char == '-') and not char == '.'] for seq in parsed_msa]
    aligned_msa = [''.join(seq) for seq in aligned_msa]

    tokenized_msa = [tokenizer.tokenizeMSA(seq) for seq in aligned_msa]
    tokenized_msa = np.array([l.tolist() for l in tokenized_msa])
    #print("TRUE IDR", row['IDR_SEQ'])
    #print("QUERY SEQUENCE", tokenizer.untokenize(tokenized_msa[query_idx]))
    #print("CAPTURED IDR", tokenizer.untokenize(tokenized_msa[query_idx, new_start_idx:new_end_idx]))
    #print("CAPTURED IDR", tokenizer.untokenize(tokenized_msa[query_idx, new_start_idx:new_end_idx]).replace("-",""))
    #import pdb; pdb.set_trace()

    msa_seq_len = len(tokenized_msa[0])
    if msa_seq_len > max_seq_len:
        # If seq len larger than max, center IDR
        idr_len = new_end_idx - new_start_idx
        buffer = int((max_seq_len - idr_len)/2)
        if new_start_idx - buffer < 0: # if IDR at beginning of seq
            print("BEGINNING")
            slice_start = 0
            slice_end = max_seq_len
            sliced_idr_start_idx = new_start_idx
            sliced_idr_end_idx = new_end_idx
        elif new_end_idx + buffer > msa_seq_len: # if IDR at end of seq
            print("END")
            slice_start = msa_seq_len - max_seq_len
            slice_end = msa_seq_len
            sliced_idr_end_idx = max_seq_len - (msa_seq_len - new_end_idx)
            sliced_idr_start_idx = sliced_idr_end_idx - idr_len
        else: # center IDR
            print("CENTER")
            slice_start = new_start_idx - buffer
            slice_end = new_end_idx + buffer
            sliced_idr_start_idx = buffer
            sliced_idr_end_idx = buffer + idr_len
        print("SLICING INDEX", slice_start, slice_end)
        print("IDR INDEX", new_start_idx, new_end_idx)
        print("IDR INDEX, adjust slice", sliced_idr_start_idx, sliced_idr_end_idx)
        #seq_len = max_seq_len
    else:
        slice_start = 0
        slice_end = msa_seq_len
        sliced_idr_start_idx = new_start_idx
        sliced_idr_end_idx = new_end_idx

    # Slice to model constraints
    sliced_msa_seq = tokenized_msa[:, slice_start: slice_end]
    #print(slice_start, slice_end)
    #print(tokenized_msa, sliced_msa_seq)
    # Remove query from array
    sliced_msa_seq = np.append(sliced_msa_seq[:query_idx], sliced_msa_seq[query_idx+1:], axis=0)
    # Query Sequence
    anchor_seq = tokenized_msa[query_idx, slice_start:slice_end]  # This is the query sequence
    #print("ANCHOR SEQ", tokenizer.untokenize(anchor_seq))
    print("VERIFY INDEXING IS CORRECT, THE FOLLOWING SHOULD MATCH")
    print("TRUE IDR", row['IDR_SEQ'])
    print("INDX IDR", tokenizer.untokenize(anchor_seq[sliced_idr_start_idx:sliced_idr_end_idx]).replace("-",""))
    #import pdb; pdb.set_trace()
    sliced_msa = [seq for seq in sliced_msa_seq if (list(set(seq)) != [tokenizer.gap_id])]
    msa_num_seqs = len(sliced_msa) + 1 # +1 accounts for query

    if msa_num_seqs > n_sequences:
        msa_n_sequences = n_sequences
        if selection_type == 'random':
            print("Using random subsampling")
            random_idx = np.random.choice(msa_num_seqs, size=n_sequences-1, replace=False)
            anchor_seq = np.expand_dims(anchor_seq, axis=0)
            output = np.concatenate((anchor_seq, np.array(sliced_msa)[random_idx.astype(int)]), axis=0)
        elif selection_type == "MaxHamming":
            print("using MaxHamming subsampling")
            output = [list(anchor_seq)]
            msa_subset = sliced_msa
            msa_ind = np.arange(msa_num_seqs-1)
            random_ind = np.random.choice(msa_ind)
            random_seq = sliced_msa[random_ind]
            output.append(list(random_seq))
            random_seq = np.expand_dims(random_seq, axis=0)
            msa_subset = np.delete(msa_subset, (random_ind), axis=0)
            m = len(msa_ind) - 1
            distance_matrix = np.ones((n_sequences - 2, m))
            #print("msa subset", msa_subset.shape, msa_num_seqs, len(msa_ind))
            #print(distance_matrix.shape)
            for i in range(n_sequences - 2):
                curr_dist = cdist(random_seq, msa_subset, metric='hamming')
                curr_dist = np.expand_dims(np.array(curr_dist), axis=0)  # shape is now (1,msa_num_seqs)
                #print(curr_dist.shape)
                distance_matrix[i] = curr_dist
                col_min = np.min(distance_matrix, axis=0)  # (1,num_choices)
                max_ind = np.argmax(col_min)
                random_ind = max_ind
                random_seq = msa_subset[random_ind]
                output.append(list(random_seq))
                random_seq = np.expand_dims(random_seq, axis=0)
                msa_subset = np.delete(msa_subset, random_ind, axis=0)
                distance_matrix = np.delete(distance_matrix, random_ind, axis=1)
    else:
        msa_n_sequences = msa_num_seqs
        output = np.full(shape=(n_sequences, max_seq_len), fill_value=tokenizer.pad_id) # TREAT SMALL SEQS as having gaps here
        output[0:1, :len(anchor_seq)] = anchor_seq
        output[1:msa_num_seqs, :len(anchor_seq)] = sliced_msa
        #output = np.concatenate((np.array(anchor_seq).reshape(1,-1), np.array(sliced_msa)), axis=0)

    output = [tokenizer.untokenize(seq) for seq in output]
    # print(len(output), len(output[0]))
    return output, sliced_idr_start_idx, sliced_idr_end_idx, msa_n_sequences

if __name__ == '__main__':
    main()
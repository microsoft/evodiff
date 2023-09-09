from evodiff.pretrained import MSA_OA_DM_MAXSUB, MSA_OA_DM_RANDSUB, ESM_MSA_1b
import numpy as np
import argparse
import torch
import os
import pickle
import evodiff
from evodiff.utils import Tokenizer, run_omegafold, clean_pdb, run_tmscore, wrap_dr_bert, read_dr_bert_output
import pathlib
from sequence_models.utils import parse_fasta
from tqdm import tqdm
import pandas as pd
import random
from evodiff.plot import aa_reconstruction_parity_plot, idr_parity_plot
from scipy.spatial.distance import hamming, cdist

def main():
    # set seeds
    _ = torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='msa_oa_dm_maxsub',
                        help='Choice of: msa_oa_dm_randsub, msa_oa_dm_maxsub, esm_msa_1b')
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
    parser.add_argument('--random-baseline', action='store_true') # for scaffold
    parser.add_argument('--scramble-baseline', action='store_true') # for IDR
    parser.add_argument('--query-only', action='store_true')
    parser.add_argument('--amlt', action='store_true')
    parser.add_argument('--single-res-domain', action='store_true', help="if start-idx = end-idx make sure to use single-res-domain flag or else you will get errors")
    args = parser.parse_args()

    if args.cond_task == 'scaffold':
        args.start_idxs.sort()
        args.end_idxs.sort()

    if args.random_baseline:
        args.model_type = 'msa_oa_dm_randsub' # placeholder

    if args.model_type == 'msa_oa_dm_randsub':
        checkpoint = MSA_OA_DM_RANDSUB()
        selection_type = 'random'
        mask_id = checkpoint[2].mask_id
        pad_id = checkpoint[2].pad_id
    elif args.model_type == 'msa_oa_dm_maxsub':
        checkpoint = MSA_OA_DM_MAXSUB()
        selection_type = 'MaxHamming'
        mask_id = checkpoint[2].mask_id
        pad_id = checkpoint[2].pad_id
    elif args.model_type == 'esm_msa_1b':
        checkpoint = ESM_MSA_1b()
        selection_type = 'MaxHamming'
        mask_id = checkpoint[2].mask_idx
        pad_id = checkpoint[2].padding_idx
    else:
        raise Exception("Please select either msa_oa_dm_randsub, msa_oa_dm_maxsub, or esm_msa_1b baseline. You selected:", args.model_type)

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
            out_fpath = home + args.model_type + '/'
        else:
            out_fpath = home + 'random-baseline/'
        if args.cond_task == 'scaffold':
            out_fpath += args.pdb + '/'
        elif args.cond_task == 'idr':
            out_fpath += 'idr/'

    if not os.path.exists(out_fpath):
        os.makedirs(out_fpath)
    if not os.path.exists(out_fpath+'plots/'):
        os.makedirs(out_fpath+'plots/')
        os.makedirs(out_fpath+'plots/svg/')

    data_top_dir = top_dir + 'data/'

    if args.cond_task == 'idr':
        if not os.path.exists(data_top_dir + 'human_idr_alignments/human_idr_boundaries_gap.tsv'):
            print("PREPROCESSING DATA")
            preprocess_IDR_data(data_top_dir)
        index_file = pd.read_csv(data_top_dir + 'human_idr_alignments/human_idr_boundaries_gap.tsv', delimiter='\t', index_col=0)
        print("INDEX FILE LEN", len(index_file))
        strings = []
        og_strings = []
        new_idrs = []
        og_idrs = []
        start_idxs = []
        end_idxs = []
        og_start_idxs = []
        og_end_idxs = []


        b_strings = []
        b_og_strings = []
        b_new_idrs = []
        b_og_idrs = []
        b_start_idxs = []
        b_end_idxs = []
        og_b_start_idxs = []
        og_b_end_idxs = []

        r_strings = []
        r_og_strings = []
        r_new_idrs = []
        r_og_idrs = []
        r_start_idxs = []
        r_end_idxs = []

        r_b_strings = []
        r_b_og_strings = []
        r_b_new_idrs = []
        r_b_og_idrs = []
        r_b_start_idxs = []
        r_b_end_idxs = []

        oma_ids = []

        for i in range(args.num_seqs):
            src, start_idx, end_idx, original_msa, num_sequences, b_src, b_start_idx, b_end_idx, oma_id = get_IDR_MSAs(index_file, data_top_dir,
                                                                                                               tokenizer,
                                                                                                               max_seq_len=args.max_seq_len,
                                                                                                               n_sequences=args.n_sequences,
                                                                                                               selection_type=selection_type,
                                                                                                               query_only=args.query_only)
            string, og_string, new_idr, og_idr, start, end = generate_idr_msa(model, original_msa, src, num_sequences, start_idx,
                                                                              end_idx, tokenizer, device=device,
                                                                              query_only=args.query_only)
            og_start, og_end = ungap_index_IDR(start, end, og_string)
            #print("before", start, end)
            start, end = ungap_index_IDR(start, end, string[0])  # Reindex start/end for ungapped seq for dr_bert analysis
            #print("after", start, end)
            #print("after", og_start, og_end)

            # print("GEN STRING", string[0].replace("-",""))
            # print("GEN STRING LEN", len(string[0].replace("-","")))
            # print("OG STRING", og_string.replace("-",""))
            # print("OG STRING LEN", len(og_string.replace("-","")))
            #print(new_idr[0].replace("-",""))
            #print(string[0].replace("-","")[start:end])
            #assert new_idr[0].replace("-","") == string[0].replace("-","")[start:end], "Generated IDR indexing wrong"
            #assert og_idr[0].replace("-","") == og_string.replace("-","")[og_start:og_end], "Original IDR indexing wrong"
            #import pdb; pdb.set_trace()
            b_string, b_og_string, b_new_idr, b_og_idr, b_start, b_end = generate_idr_msa(model, original_msa, b_src, num_sequences, b_start_idx,
                                                                             b_end_idx, tokenizer, device=device,
                                                                             query_only=args.query_only)
            og_b_start, og_b_end = ungap_index_IDR(b_start, b_end, b_og_string)
            b_start, b_end = ungap_index_IDR(b_start, b_end, b_string[0])

            if args.scramble_baseline:
                r_string, r_og_string, r_new_idr, r_og_idr, r_start, r_end = scramble_query(original_msa, start_idx, end_idx)
                r_b_string, r_b_og_string, r_b_new_idr, r_b_og_idr, r_b_start, r_b_end = scramble_query(original_msa, b_start_idx, b_end_idx)

            else:
                r_string, r_og_string, r_new_idr, r_og_idr, r_start, r_end = generate_idr_msa(model, original_msa, src, num_sequences, start_idx,
                                                                              end_idx, tokenizer, device=device,
                                                                              query_only=args.query_only, random_baseline=True,
                                                                            data_top_dir=data_top_dir)
                r_b_string, r_b_og_string, r_b_new_idr, r_b_og_idr, r_b_start, r_b_end = generate_idr_msa(model, original_msa, b_src, num_sequences,
                                                                                          b_start_idx,
                                                                                          b_end_idx, tokenizer,
                                                                                          device=device,
                                                                                          query_only=args.query_only, random_baseline=True,
                                                                            data_top_dir=data_top_dir)
            r_start, r_end = ungap_index_IDR(r_start, r_end, r_string[0])
            r_b_start, r_b_end = ungap_index_IDR(r_b_start, r_b_end-1, r_b_string[0])

            oma_ids.append(oma_id)

            strings.append(string)
            og_strings.append(og_string)
            new_idrs.append(new_idr)
            og_idrs.append(og_idr)
            start_idxs.append(start)
            end_idxs.append(end)
            og_start_idxs.append(og_start)
            og_end_idxs.append(og_end)

            b_strings.append(b_string)
            b_og_strings.append(b_og_string)
            b_new_idrs.append(b_new_idr)
            b_og_idrs.append(b_og_idr)
            b_start_idxs.append(b_start)
            b_end_idxs.append(b_end)
            og_b_start_idxs.append(og_b_start)
            og_b_end_idxs.append(og_b_end)

            r_strings.append(r_string)
            r_og_strings.append(r_og_string)
            r_new_idrs.append(r_new_idr)
            r_og_idrs.append(r_og_idr)
            r_start_idxs.append(r_start)
            r_end_idxs.append(r_end)

            r_b_strings.append(r_b_string)
            r_b_og_strings.append(r_b_og_string)
            r_b_new_idrs.append(r_b_new_idr)
            r_b_og_idrs.append(r_b_og_idr)
            r_b_start_idxs.append(r_b_start)
            r_b_end_idxs.append(r_b_end)

        with open(out_fpath + 'original_samples_string.fasta', 'w') as f:
            for i, _s in enumerate(og_strings):
                f.write(">SEQUENCE_" + str(i) + "\n" + str(_s) + "\n")
        save_df = pd.DataFrame(list(zip(new_idrs, og_idrs, start_idxs, end_idxs)),
                               columns=['gen_idrs', 'original_idrs', 'start_idxs', 'end_idxs'])
        save_df.to_csv(out_fpath + 'idr_df.csv', index=True)
        og_save_df = pd.DataFrame(list(zip(og_idrs, og_idrs, og_start_idxs, og_end_idxs)),
                               columns=['gen_idrs', 'original_idrs', 'start_idxs', 'end_idxs'])
        og_save_df.to_csv(out_fpath + 'og_idr_df.csv', index=True)
        b_save_df = pd.DataFrame(list(zip(b_new_idrs, b_og_idrs, b_start_idxs, b_end_idxs)),
                               columns=['gen_idrs', 'original_idrs', 'start_idxs', 'end_idxs'])
        b_save_df.to_csv(out_fpath + 'og_baseline_idr_df.csv', index=True)
        og_b_save_df = pd.DataFrame(list(zip(new_idrs, og_idrs, og_b_start_idxs, og_b_end_idxs)),
                                  columns=['gen_idrs', 'original_idrs', 'start_idxs', 'end_idxs'])
        og_b_save_df.to_csv(out_fpath + 'idr_df.csv', index=True)

        # Write OMA_ID to file for reference
        with open(out_fpath + 'queried_ids.csv', 'w') as f:
            [f.write(o_id + "\n") for o_id in oma_ids]
        f.close()


    elif args.cond_task == 'scaffold':
        strings = []
        start_idxs = []
        end_idxs = []
        scaffold_lengths = []
        for i in range(args.num_seqs): # no batching
            print("SEQ", i)
            motif_start_idxs = args.start_idxs
            motif_end_idxs = [i + 1 for i in args.end_idxs]  # inclusive of final residue
            # 50/50 split on MSAs
            if i <50:
                selection_type='random'
            else:
                selection_type='MaxHamming'
            sliced_msa, sliced_start_idxs, sliced_end_idxs, original_motif = subsample_MSA(i, data_top_dir, args.pdb,
                                                                                           motif_start_idxs,
                                                                                           motif_end_idxs,
                                                                                           Tokenizer(), query_idx=0,
                                                                                           max_seq_len=args.max_seq_len,
                                                                                           n_sequences=args.n_sequences,
                                                                                           selection_type=selection_type)
            string, new_start_idx, new_end_idx, seq_len = generate_scaffold_msa(args.model_type, model, sliced_msa,
                                                                                sliced_start_idxs, sliced_end_idxs,
                                                                       data_top_dir, tokenizer, device=device,
                                                                       random_baseline=args.random_baseline,
                                                                       query_only=args.query_only,
                                                                       n_sequences=args.n_sequences,
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
    f.close()

    with open(out_fpath + 'generated_samples_string.fasta', 'w') as f:
        for i, _s in enumerate(strings):
            f.write(">SEQUENCE_" + str(i) + "\n" + str(_s[0]) + "\n")
    f.close()

    if args.cond_task == 'idr':
        with open(out_fpath + 'baseline_samples_string.fasta', 'w') as f:
            for i, _s in enumerate(b_strings):
                f.write(">SEQUENCE_" + str(i) + "\n" + str(_s[0]) + "\n")
        f.close()
        with open(out_fpath + 'random_baseline_samples_string.fasta', 'w') as f:
            for i, _s in enumerate(r_b_strings):
                f.write(">SEQUENCE_" + str(i) + "\n" + str(_s[0]) + "\n")
        f.close()
        with open(out_fpath + 'random_generated_samples_string.fasta', 'w') as f:
            for i, _s in enumerate(r_strings):
                f.write(">SEQUENCE_" + str(i) + "\n" + str(_s[0]) + "\n")
        f.close()

        wrap_dr_bert(out_fpath, generated_fasta_file='generated_samples_string.fasta', path_to_dr_bert=top_dir + '../DR-BERT/', out_file='gen_out.pkl')
        wrap_dr_bert(out_fpath, generated_fasta_file='original_samples_string.fasta', path_to_dr_bert=top_dir + '../DR-BERT/', out_file='og_out.pkl')
        wrap_dr_bert(out_fpath, generated_fasta_file='baseline_samples_string.fasta', path_to_dr_bert=top_dir + '../DR-BERT/', out_file='b_out.pkl')
        wrap_dr_bert(out_fpath, generated_fasta_file='random_baseline_samples_string.fasta', path_to_dr_bert=top_dir + '../DR-BERT/', out_file='r_b_out.pkl')
        wrap_dr_bert(out_fpath, generated_fasta_file='random_generated_samples_string.fasta', path_to_dr_bert=top_dir + '../DR-BERT/', out_file='r_gen_out.pkl')

        true_disorder_score, true_order_score = evodiff.utils.read_dr_bert_output(out_fpath, 'true', out_fpath + 'og_out.pkl', out_fpath + 'og_out.pkl', og_save_df, og_b_save_df)
        gen_disorder_score, gen_order_score = evodiff.utils.read_dr_bert_output(out_fpath, 'gen', out_fpath + 'gen_out.pkl', out_fpath + 'b_out.pkl', save_df, b_save_df)
        random_disorder_score, random_order_score = evodiff.utils.read_dr_bert_output(out_fpath, 'random', out_fpath + 'r_gen_out.pkl', out_fpath + 'r_b_out.pkl', save_df, b_save_df)

        plot_df = pd.DataFrame({'score': true_disorder_score, 'region': ["disorder"]*len(true_disorder_score), 'type': ["true"]*len(true_disorder_score)})
        #print(plot_df)
        plot_df = plot_df.append(pd.DataFrame({'score': true_order_score, 'region': ["order"]*len(true_order_score), 'type': ["true"]*len(true_order_score)}), ignore_index=True)
        #print(plot_df)
        plot_df = plot_df.append(pd.DataFrame({'score': gen_disorder_score, 'region': ["disorder"]*len(gen_disorder_score), 'type': ["gen"]*len(gen_disorder_score)}), ignore_index=True)
        plot_df = plot_df.append(pd.DataFrame({'score': gen_order_score, 'region': ["order"]*len(gen_order_score), 'type': ["gen"]*len(gen_order_score)}), ignore_index=True)
        plot_df = plot_df.append(pd.DataFrame(
            {'score': random_disorder_score, 'region': ["disorder"] * len(random_disorder_score),
             'type': ["random"] * len(random_disorder_score)}), ignore_index=True)
        plot_df = plot_df.append(
            pd.DataFrame({'score': random_order_score, 'region': ["order"] * len(random_order_score),
                          'type': ["random"] * len(random_order_score)}), ignore_index=True)

        plot_df.to_csv(out_fpath + 'drbert_scores_df.csv', index=True)

        evodiff.plot.idr_boxplot_all(plot_df, out_fpath+'plots/', save_name='combined_')
        evodiff.plot.idr_boxplot(true_disorder_score, true_order_score, out_fpath + 'plots/', save_name='true_')
        evodiff.plot.idr_boxplot(gen_disorder_score, gen_order_score, out_fpath + 'plots/', save_name='gen_')
        evodiff.plot.idr_boxplot(random_disorder_score, random_order_score, out_fpath + 'plots/', save_name='random_')
    elif args.cond_task == 'scaffold':
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

def subsample_MSA(save_idx, data_top_dir, pdb, start_idxs, end_idxs, tokenizer, query_idx=0, max_seq_len=512, n_sequences=64, selection_type='random'):
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
    save_path = data_top_dir + '/scaffolding-msas/' + pdb + '/'
    msa_save_file = save_path + pdb + '_' + str(save_idx) + '.a3m'
    tokenized_msa = get_MSA(data_top_dir + '/scaffolding-msas/' + pdb + '.a3m', tokenizer=Tokenizer())
    original_motif = [tokenizer.untokenize(tokenized_msa[0][start_idxs[i]:end_idxs[i]]) for i in range(len(start_idxs))]
    print("ORIGINAL MOTIF", original_motif)

    if  os.path.isfile(msa_save_file):
        output= get_MSA(msa_save_file, tokenizer=Tokenizer())
        output = [tokenizer.untokenize(seq) for seq in output]
        #print(output)
        with open(save_path + 'start_idxs_'+str(save_idx) + '.pkl', "rb") as fp:
            sliced_start_idxs = pickle.load(fp)
        with open(save_path + 'end_idxs_'+str(save_idx) + '.pkl', "rb") as fp:
            sliced_end_idxs = pickle.load(fp)
    else:
        # Else sample MSA and save to file:
        if not os.path.exists(data_top_dir + '/scaffolding-msas/' + pdb + '/'):
            os.mkdir(data_top_dir + '/scaffolding-msas/' + pdb + '/')

        msa_seq_len = len(tokenized_msa[0])
        motif_start = start_idxs[0]
        motif_end = end_idxs[-1]

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

        with open(msa_save_file, 'a') as f:
            for seq_num in range(len(output)):
                seq_string = str(output[seq_num]).replace('!', '')  # remove PADs
                if seq_num == 0 :
                    f.write(">MSA_0" + "\n" + str(seq_string) + "\n")
                else:
                    f.write(">tr \n" + str(seq_string) + "\n" )
            f.close()
        with open(save_path + 'start_idxs_'+str(save_idx) + '.pkl', "wb") as fp:
            pickle.dump(sliced_start_idxs, fp)
        with open(save_path + 'end_idxs_'+str(save_idx) + '.pkl', "wb") as fp:
            pickle.dump(sliced_end_idxs, fp)
    return output, sliced_start_idxs, sliced_end_idxs, original_motif

def get_masked_locations(query_sequence, sliced_start_idxs, sliced_end_idxs, pad_id):
    "Return list of masked indices given a list of starting and ending indeces for motifs"
    #input_mask = (query_sequence != pad_id)
    seq_len = len(query_sequence)
    all_index = np.arange(seq_len)
    list_motif = [list(range(sliced_start_idxs[i], sliced_end_idxs[i])) for i in range(len(sliced_start_idxs))]
    list_motif = [item for sublist in list_motif for item in sublist]
    list_masked = [x for x in all_index if x not in list_motif]
    #print(list_masked)
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
    if model_type == 'msa_oa_dm_maxsub' or model_type == 'msa_oa_dm_randsub':
        return [tokenizer.tokenizeMSA(seq) for seq in untokenized]
    elif model_type == 'esm_msa_1b':
        src = []
        for i, seq in enumerate(untokenized):
            new_seq = [tokenizer.cls_idx] + [tokenizer.get_idx(c) for c in [*seq]] + [tokenizer.eos_idx]
            src.append(new_seq)
        return src

def untokenize_msa(model_type, tokenized, tokenizer):
    if model_type == 'msa_oa_dm_maxsub' or model_type == 'msa_oa_dm_randsub':
        return tokenizer.untokenize(tokenized)
    elif model_type == 'esm_msa_1b':
        return ''.join([tokenizer.get_tok(s) for s in tokenized[1:-1]])


def generate_scaffold_msa(model_type, model, sliced_msa, sliced_start_idxs, sliced_end_idxs, data_top_dir, tokenizer, query_only=True,
                      device='gpu', random_baseline=False, n_sequences=64,
                      mask=0, pad=1):
    #motif_end_idxs = [i + 1 for i in motif_end_idxs]  # inclusive of final residue
    if random_baseline:
        train_prob_dist = aa_reconstruction_parity_plot(data_top_dir+'../', 'reference/', 'placeholder.csv', gen_file=False)

    # tokenized_msa = get_MSA(data_top_dir + '/scaffolding-msas/' + PDB_ID+'.a3m', tokenizer=Tokenizer())
    # sliced_msa, sliced_start_idxs, sliced_end_idxs, original_motif = subsample_MSA(tokenized_msa, motif_start_idxs, motif_end_idxs,
    #                                                              Tokenizer(), query_idx=0, max_seq_len=max_seq_len,
    #                                                              n_sequences=n_sequences, selection_type=selection_type)

    print("INPUT MSA", sliced_msa[0])

    # Now tokenize using tokenizer of choice
    sliced_msa = tokenize_msa(model_type, sliced_msa, tokenizer)
    query_sequence = sliced_msa[0]  # ensure query is first seq -> not true for IDRs

    if model_type == 'esm_msa_1b':
        seq_len = len(query_sequence)-2
        mask_locations = get_masked_locations(query_sequence[1:-1], sliced_start_idxs, sliced_end_idxs, pad_id=pad)
        mask_locations = [i + 1 for i in mask_locations]
        max_token = len(tokenizer)
        x_token_location = tokenizer.get_idx('X')
    else:
        seq_len = len(query_sequence)
        mask_locations = get_masked_locations(query_sequence, sliced_start_idxs, sliced_end_idxs, pad_id=pad)
        max_token = tokenizer.K - 1
        x_token_location = tokenizer.tokenize('X')
    print("X TOKEN IDX", x_token_location)

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
                # # # Penalize X token
                # penalty = torch.ones(p_softmax.shape).cuda()
                # penalty[:, x_token_location] += 100
                # p_softmax /= penalty
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
            # # Penalize X token
            penalty = torch.ones(p_softmax.shape)
            penalty = penalty.to(device)
            penalty[:, x_token_location] += 100
            p_softmax /= penalty
            p_sample = torch.multinomial(input=p_softmax, num_samples=1)
            p_sample = p_sample.squeeze()
            sample[:, random_x, random_y] = p_sample
            #print(untokenize_msa(model_type, sample[0][0], tokenizer))

    untokenized = [untokenize_msa(model_type, sample[0][0], tokenizer)] # only return query sequence
    print(untokenized)

    return untokenized, sliced_start_idxs, [i - 1 for i in sliced_end_idxs], seq_len  # return output and untokenized output, re-indexed motif starts and ends (ends-1 for rmsd analyis)

def scramble_query(original_msa, start_idx, end_idx):
    scrambled_seqs = []

    original_idr = original_msa[0][start_idx:end_idx]
    scrambled_idr = list(original_idr)
    np.random.shuffle(scrambled_idr)
    scrambled_idr = ''.join(scrambled_idr)

    print("original_idr", original_idr)
    print("scrambled_idr", scrambled_idr)

    scrambled_sequence = [original_msa[0][:start_idx] + scrambled_idr + original_msa[0][end_idx:]]
    # print("full sequence", scrambled_sequence)
    print(len(scrambled_sequence[0]), len(original_msa[0]))
    assert len(scrambled_sequence[0]) == len(original_msa[0]), "SCRAMBLED seq different length"

    return scrambled_sequence, original_msa[0], scrambled_idr, original_idr, start_idx, end_idx


def generate_idr_msa(model, original_msa, src, num_sequences, start_idx, end_idx, tokenizer, device='gpu', query_only=True, random_baseline=False, data_top_dir='data/'):
    src = torch.tensor(src).unsqueeze(0) # Make batchsize 1
    if random_baseline:
        train_prob_dist = aa_reconstruction_parity_plot(data_top_dir+'../', 'reference/', 'placeholder.csv', gen_file=False)
    if query_only:
        masked_loc_x = [0]
    else:
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
            if random_baseline:
                p_sample = torch.multinomial(torch.tensor(train_prob_dist), num_samples=1)
            else:
                #print(sample.shape)
                preds = model(sample)  # Output shape of preds is (BS=1, N=64, L, n_tokens=31)
                #print("preds", preds.shape)
                #print(random_x, random_y)
                p = preds[:, random_x, random_y, :]
                # if random_x == 0:  # for first row don't let p_softmax predict gaps
                #     p = preds[:, random_x, random_y, :tokenizer.K - 1]
                p_softmax = torch.nn.functional.softmax(p, dim=1)
                # Penalize gaps
                #penalty = torch.ones(p.shape).to(p.device)
                #penalty[:, -1] += penalty_value
                # print(p_softmax)
                #p_softmax /= penalty
                # print(p_softmax)
                p_sample = torch.multinomial(input=p_softmax, num_samples=1)
                p_sample = p_sample.squeeze()
            sample[:, random_x, random_y] = p_sample
            #print(tokenizer.untokenize(sample[0][0][start_idx:end_idx]))
    #print(sample.shape)
    #print([tokenizer.untokenize(seq) for seq in sample[0]])
    new_idr = [tokenizer.untokenize(sample[0][0][start_idx:end_idx])]
    untokenized_query_msa = [tokenizer.untokenize(sample[0][0])]
    og_idr = [original_msa[0][start_idx:end_idx]]

    # print("NEW_IDR", new_idr)
    # print("UNTOKENIZED", untokenized_query_msa)
    # print("OG IDR", og_idr)
    # print("OG SEQ", original_msa[0])
    # import pdb; pdb.set_trace()
    return untokenized_query_msa, original_msa[0], new_idr, og_idr, start_idx, end_idx  # return gen_query, og_query, new_idrs, og_idrs

def mask_idr(seq, new_start_idx, new_end_idx, i, num_unpadded_rows):
    if i < num_unpadded_rows:
        idr_range = new_end_idx - new_start_idx
        masked_seq = seq[0:new_start_idx] + '#' * idr_range + seq[new_end_idx:]
    else:
        masked_seq = seq
    return masked_seq

def reindex_IDR(start_idx, end_idx, query_seq, gapped_query_seq):
    """
    From a start and end idx corresponding to an ungapped sequence, get the start and end idx for a gapped sequence
    """
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
    new_start = offset[old_start] + start_idx # original idx starts at 1
    new_end = offset[old_end] + end_idx

    return new_start, new_end  # new range of IDR (inclusive)

def ungap_index_IDR(start_gapped, end_gapped, gapped_query_seq):
    """
    From a start and end idx corresponding to an gapped sequence, get the start and end idx for an ungapped sequence
    """
    gap_count = 0
    offset = []  # This tracks how many gaps between letters
    for aa in list(gapped_query_seq):
        if aa == '-':
            offset.append(gap_count)
            gap_count += 1
        else:
            offset.append(gap_count)
    #print(len(offset), len(gapped_query_seq))
    assert len(offset) == len(gapped_query_seq)

    # Add gaps to old index to get new start/end index
    # if offset[start_gapped] == 0 or offset[end_gapped] == 0:
    print(gapped_query_seq)
    print(offset)
    print("minus", offset[start_gapped])
    print("minus", offset[end_gapped-1])
    #import pdb; pdb.set_trace()
    start_ungapped = start_gapped - offset[start_gapped]
    end_ungapped = end_gapped - offset[end_gapped-1]

    return start_ungapped, end_ungapped  # new range of IDR (inclusive)

def preprocess_IDR_data(data_top_dir):
    data_dir = data_top_dir + 'human_idr_alignments/'
    all_files = os.listdir(data_dir + 'human_protein_alignments')
    index_file = pd.read_csv(data_dir + 'human_idr_boundaries.tsv', delimiter='\t')

    # Filter out IDRs > 250 residues in length
    # index_file['LENGTHS'] = list(index_file['END'] - index_file['START'])
    print("BEFORE", len(index_file))
    # index_file = index_file[index_file['LENGTHS'] <= 250]
    # print("AFTER FILTERING LONG IDRS", len(index_file))
    # # print(index_file.head())
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
    #index_file = index_file[index_file['GAP LENGTHS'] <= 250]
    print("AFTER", len(index_file))
    index_file.to_csv(data_dir + 'human_idr_boundaries_gap.tsv', sep='\t')

def get_IDR_MSAs(index_file, data_top_dir, tokenizer, max_seq_len=512, n_sequences=64, selection_type='random', query_only=True):
    # GET IDRS
    # index = random.randint(0, len(index_file) - 1)
    #
    # data_dir = data_top_dir + 'human_idr_alignments/'
    # all_files = os.listdir(data_dir + 'human_protein_alignments')
    # if not os.path.exists(data_dir + 'human_idr_boundaries_gap.tsv'):
    #     preprocess_IDR_data(data_top_dir)
    # print("USING INDEX", index)
    # row = index_file.iloc[index]
    # # Get MSA
    # msa_file = [file for i, file in enumerate(all_files) if row['OMA_ID'] in file][0]
    msa_data, new_start_idx, new_end_idx, num_sequences, b_start_idx, b_end_idx, oma_id = subsample_IDR_MSA(index_file, tokenizer, max_seq_len=max_seq_len, n_sequences=n_sequences,
                                 selection_type=selection_type, data_top_dir=data_top_dir)
    print(len(msa_data[0]))
    # MASK out IDR
    masked_msa = msa_data.copy()
    masked_msa[0] = mask_idr(msa_data[0], new_start_idx, new_end_idx, 0, num_sequences)
    print(len(masked_msa[0]))
    if not query_only:
        masked_msa = [mask_idr(seq, new_start_idx, new_end_idx, i, num_sequences) for i, seq in enumerate(msa_data)]

    # MASK out non-IDR baseline
    b_masked_msa = msa_data.copy()
    b_masked_msa[0] = mask_idr(msa_data[0], b_start_idx, b_end_idx, 0, num_sequences)
    print(len(b_masked_msa[0]))
    print(b_masked_msa[0])
    if not query_only:
        b_masked_msa = [mask_idr(seq, b_start_idx, b_end_idx, i, num_sequences) for i, seq in enumerate(msa_data)]

    tokenized_msa = [tokenizer.tokenizeMSA(seq) for seq in masked_msa]
    tokenized_msa = np.array([l.tolist() for l in tokenized_msa])
    b_tokenized_msa = [tokenizer.tokenizeMSA(seq) for seq in b_masked_msa]
    print("sEQUENCE")
    print("LENGTH", len(b_tokenized_msa))
    print("FIRST SEQ LEN", len(b_tokenized_msa[0]))
    print([len(l) for l in b_tokenized_msa if len(l) != len(b_tokenized_msa[0])])
    b_tokenized_msa = np.array([l.tolist() for l in b_tokenized_msa])

    return tokenized_msa, new_start_idx, new_end_idx, msa_data, num_sequences, b_tokenized_msa, b_start_idx, b_end_idx, oma_id

import itertools
def intervals_extract(iterable):
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable),
                                        lambda t: t[1] - t[0]):
        group = list(group)
        yield [group[0][1], group[-1][1]]

def subsample_IDR_MSA(index_file, tokenizer, max_seq_len=512, n_sequences=64, selection_type='random', data_top_dir='data/'):
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
    data_dir = data_top_dir + 'human_idr_alignments/'
    all_files = os.listdir(data_dir + 'human_protein_alignments')
    count = 0

    for i in range(len(index_file)):
        while count < 1:
            index = random.randint(0, len(index_file) - 1) #2 #13160 #
            row = index_file.loc[index]
            # Get MSA
            msa_file = [file for i, file in enumerate(all_files) if row['OMA_ID'] in file][0]
            parsed_msa, msa_names = parse_fasta(data_dir + 'human_protein_alignments/' + msa_file, return_names=True)
            # Get query
            query_idx = [i for i, name in enumerate(msa_names) if name == row['OMA_ID']][0]  # get query index

            new_start_idx = row['GAP START']
            new_end_idx = row['GAP END']

            aligned_msa = [[char for char in seq if (char.isupper() or char == '-') and not char == '.'] for seq in
                           parsed_msa]
            aligned_msa = [''.join(seq) for seq in aligned_msa]

            tokenized_msa = [tokenizer.tokenizeMSA(seq) for seq in aligned_msa]
            tokenized_msa = np.array([l.tolist() for l in tokenized_msa])

            msa_seq_len = len(tokenized_msa[0])
            idr_len = new_end_idx - new_start_idx #len(row['IDR_SEQ']) #
            idr_len_nogaps = len(row['IDR_SEQ']) #new_end_idx - new_start_idx

            #TESTING
            # These two lines should be the same thing
            # print("TRUE IDR", row['IDR_SEQ'])
            # print(tokenizer.untokenize(tokenized_msa[query_idx, new_start_idx:new_end_idx]).replace("-", ""))
            # print(idr_len)
            # print(idr_len_nogaps)
            # import pdb; pdb.set_trace()

            # Get non-idr ranges
            query_rows = index_file[index_file["OMA_ID"] == row['OMA_ID']]
            idr_ranges = []
            for i in range(len(query_rows)):
                idr_range = np.arange(query_rows.iloc[i]['GAP START'], query_rows.iloc[i]['GAP END'])
                idr_ranges.extend(idr_range)
            # print(idr_ranges)
            seq_indices = np.arange(0, msa_seq_len)
            non_idr_indices = [s for s in seq_indices if s not in idr_ranges]
            non_idr_ranges = list(intervals_extract(non_idr_indices))
            # print(non_idr_ranges)
            non_idr_ranges = [r for r in non_idr_ranges if r[1] - r[0] > idr_len_nogaps]

            # Subsample MSA of max_seq_len around IDR indices, inclusive of non-IDR baseline region
            if msa_seq_len > max_seq_len and idr_len < (max_seq_len / 2) and len(non_idr_ranges) > 0:
                # If seq len larger than max, idr is less than half the sequence, and there are non-idr ranges in sequence center IDR
                buffer = int((max_seq_len - idr_len) / 2)
                if new_start_idx - buffer < 0:  # if IDR at beginning of seq
                    print("BEGINNING")
                    slice_start = 0
                    slice_end = max_seq_len
                    sliced_idr_start_idx = new_start_idx
                    sliced_idr_end_idx = new_end_idx
                    b_start_idx = sliced_idr_end_idx + 1
                    if b_start_idx + idr_len_nogaps >= max_seq_len: # Reduce baseline length if longer than max len
                        b_end_idx = max_seq_len
                    else:
                        b_end_idx = b_start_idx + idr_len_nogaps
                elif new_end_idx + buffer > msa_seq_len:  # if IDR at end of seq
                    print("END")
                    slice_start = msa_seq_len - max_seq_len
                    slice_end = msa_seq_len
                    sliced_idr_end_idx = max_seq_len - (msa_seq_len-new_end_idx)
                    sliced_idr_start_idx = sliced_idr_end_idx - idr_len
                    b_end_idx = sliced_idr_start_idx - 1
                    b_start_idx = b_end_idx - idr_len_nogaps
                    if b_start_idx - idr_len_nogaps < 0 : # Reduce baseline length if longer than max len
                        b_start_idx = 0
                    else:
                        b_end_idx = b_start_idx + idr_len_nogaps
                else:  # center IDR
                    print("CENTER")
                    slice_start = new_start_idx - buffer
                    slice_end = new_end_idx + buffer
                    sliced_idr_start_idx = buffer
                    sliced_idr_end_idx = sliced_idr_start_idx + idr_len
                    b_start_idx = sliced_idr_end_idx + 1
                    if b_start_idx + idr_len_nogaps >= max_seq_len: # Reduce baseline length if longer than max len
                        b_end_idx = max_seq_len
                    else:
                        b_end_idx = b_start_idx + idr_len_nogaps
                print("SLICING INDEX", slice_start, slice_end)
                print("BASELINE INDEX", b_start_idx, b_end_idx)
                if slice_end-slice_start < max_seq_len: # If slicing sequence length 511 in center, correct length
                    slice_end += max_seq_len-(slice_end-slice_start)
                    print("ADJUST SLICING INDEX", slice_start, slice_end)
                print("IDR INDEX, adjust slice", sliced_idr_start_idx, sliced_idr_end_idx)
                # seq_len = max_seq_len
                count += 1
            elif msa_seq_len < max_seq_len and idr_len < msa_seq_len / 2 and len(non_idr_ranges) > 0:
                print("SHORT SEQ")
                slice_start = 0
                slice_end = msa_seq_len
                sliced_idr_start_idx = new_start_idx
                sliced_idr_end_idx = new_end_idx
                b_start_idx = non_idr_ranges[0][0]
                b_end_idx = b_start_idx + idr_len_nogaps
                count += 1
            else:
                print("SKIPPING MSA")
        else:
            break

        if b_start_idx in idr_ranges or b_end_idx in idr_ranges:
            print("BASELINE INDICES ARE IN IDR RANGES")
            import pdb; pdb.set_trace()

    # Slice to model constraints
    sliced_msa_seq = tokenized_msa[:, slice_start: slice_end]
    print("SLICED MSA_SEQ", len(sliced_msa_seq[0]), len(sliced_msa_seq[1]))
    #print(slice_start, slice_end)
    #print(tokenized_msa, sliced_msa_seq)
    # Remove query from array
    sliced_msa_seq = np.append(sliced_msa_seq[:query_idx], sliced_msa_seq[query_idx+1:], axis=0)
    # Query Sequence
    anchor_seq = tokenized_msa[query_idx, slice_start:slice_end]  # This is the query sequence
    print("ANCHOR SEQ", len(tokenizer.untokenize(anchor_seq)),  tokenizer.untokenize(anchor_seq))
    # TODO: what is going on here?
    print("VERIFY INDEXING IS CORRECT, THE FOLLOWING SHOULD MATCH")
    print("SAMPLING INDEX", index)
    # print("IDR LEN", idr_len)
    # print("IDR LEN NO GAPS", idr_len_nogaps)
    print("TRUE IDR:", row['IDR_SEQ'])
    # print("LEN, START, END", len(anchor_seq), sliced_idr_start_idx, sliced_idr_end_idx )
    print("INDX IDR:", tokenizer.untokenize(anchor_seq[sliced_idr_start_idx:sliced_idr_end_idx]).replace("-", ""))
    print("BASELINE SEQ", tokenizer.untokenize(anchor_seq[b_start_idx:b_end_idx]).replace("-", ""))
    #print("LENGTHS", idr_len_nogaps, len(tokenizer.untokenize(anchor_seq[b_start_idx:b_end_idx])))
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
        print("N_SEQ < MSA SEQUENCES")
        msa_n_sequences = msa_num_seqs
        output = np.full(shape=(n_sequences, max_seq_len), fill_value=tokenizer.pad_id)
        output[0:1, :len(anchor_seq)] = anchor_seq
        output[1:msa_num_seqs, :len(anchor_seq)] = sliced_msa
        #output = np.concatenate((np.array(anchor_seq).reshape(1,-1), np.array(sliced_msa)), axis=0)

    output = [tokenizer.untokenize(seq) for seq in output]
    #print("FINAL LENS", len(output[0]))
    return output, sliced_idr_start_idx, sliced_idr_end_idx, msa_n_sequences, b_start_idx, b_end_idx, row['OMA_ID']

if __name__ == '__main__':
    main()
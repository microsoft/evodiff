import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from analysis.plot import plot_tmscore
import argparse
import difflib
from ast import literal_eval
from Bio.PDB import PDBParser, Selection
import esm.inverse_folding

def main():
    # set seeds
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='oa_ar_640M',
                        help='Choice of: carp_38M carp_640M esm1b_650M \
                                  oa_ar_38M oa_ar_640M lr_ar_38M lr_ar_640M')
    parser.add_argument('--pdb', type=str, default=None,
                        help="If using cond-task=scaffold, provide a PDB code and motif indexes")
    parser.add_argument('--start-idxs', type=int, action='append',
                        help="If using cond-task=scaffold, provide start and end indexes for motif being scaffolded\
                                 If defining multiple motifs, supply the start and end -idx motif as a new argument\
                                  ex: --start-idx 3 --end-idx 10 --start-idx 20 --end-idx 25\
                                  indexes are inclusive of both start and end values.\
                                  WARNING: PDBs are OFTEN indexed at a number that is not 1. If your PDB file begins at 4\
                                  and the motif you want to query is residues 5 to 10, as defined by the PDB, your inputs to\
                                  this code should be --start-idx 1 and --end-idx 6\
                                  WARNING: Motifs start/end cannot overap in regions.\
                                  entry: --start-idx 5 end-idx 10 --start-idx 6 end-idx 11 is NOT valid\
                                  instead use: --start-idx 5--end-idx 11")
    parser.add_argument('--end-idxs', type=int, action='append')
    parser.add_argument('--chain', type=str, default='A',
                        help="chain in PDB")
    parser.add_argument('--num-seqs', type=int, default=10,
                        help="Number of sequences generated per scaffold length")
    parser.add_argument('--scaffold-min', type=int, default=1,
                        help="Min scaffold len ")
    parser.add_argument('--scaffold-max', type=int, default=30,
                        help="Max scaffold len, will randomly choose a value between min/max")
    args = parser.parse_args()

    home = 'cond-gen/' + args.model_type + '/'

    args.start_idxs.sort()
    args.end_idxs.sort()
    print(args.start_idxs, args.end_idxs)

    ref_pdb = args.pdb + '_reres.pdb'
    # Iterate over all generated files and calc rmsd
    motif_df = calc_rmsd(args.num_seqs, ref_pdb, fpath=home + args.pdb, ref_motif_starts=args.start_idxs,
                         ref_motif_ends=args.end_idxs)
    ci_scores, ci_sampled, ci_fixed = get_confidence_score(home + args.pdb, args.num_seqs,  motif_df)
    motif_df['scores'] = ci_scores
    motif_df['scores_sampled'] = ci_sampled
    motif_df['scores_fixed'] = ci_fixed
    motif_df_sorted = motif_df.sort_values('scores_fixed', ascending=False)
    print(motif_df_sorted[['seqs','scores','scores_sampled','scores_fixed','rmsd']])
    #print(len(motif_df_sorted[['seqs', 'scores', 'scores_sampled', 'scores_fixed', 'rmsd']]))
    candidates = motif_df_sorted[motif_df_sorted['rmsd'] <= 1.5]
    print(candidates[['seqs','scores','scores_sampled','scores_fixed','rmsd']])
    print("Success:", len(candidates))
    # for seq in motif_df[motif_df['rmsd'] <= 3.5]['seqs'][:10]:
    #     print(seq[1:-1])
    motif_df.to_csv(home + args.pdb + '/motif_df_rmsd.csv', index=True)

    # Uncomment if you also want to look at TM scores (need to run first)
    tm = pd.read_csv(home + args.pdb + '/tmscores.txt', names=['tmscore'])
    plot_tmscore(tm, ['grey'], legend=False, save_file=args.pdb+'_tmscore')

    # Plot rmsd vs plddt
    plot_rmsd(args.pdb, motif_df)

    # percent similarity in fixed region
    chain_ids=[args.chain]
    structure = esm.inverse_folding.util.load_structure(home+args.pdb+'/pdb/'+ref_pdb, chain_ids)
    coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
    sequence = native_seqs[chain_ids[0]]
    original_fixed = sequence[args.start_idxs[0]:args.end_idxs[-1]]
    sim = []
    for i in range(len(motif_df)):
        new_motif_starts = literal_eval(motif_df['start_idxs'].iloc[i])[0]
        new_motif_ends = literal_eval(motif_df['end_idxs'].iloc[i])[-1]
        gen_sequence = motif_df['seqs'].iloc[i][2:-2]
        #print(gen_sequence)
        gen_fixed = gen_sequence[new_motif_starts:new_motif_ends]
        #print("original fixed domain", original_fixed)
        #print("new domain", gen_fixed)
        sim.append(calc_sim(original_fixed, gen_fixed))
        #print(sim)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    sns.histplot(sim, color='grey', bins=10, ax=ax)
    plt.xlabel('% Seq similarity (Fixed)')
    plt.title("  ")
    plt.xlim(0, 100)
    plt.tight_layout()
    fig.savefig(os.path.join('plots/' + args.pdb + '_similarity.png'))


def calc_sim(seq1, seq2):
    sm=difflib.SequenceMatcher(None,seq1,seq2)
    sim = sm.ratio()*100
    return sim

def calc_rmsd(num_structures, reference_PDB, fpath='conda/gen/6exz', ref_motif_starts=[30], ref_motif_ends=[44]):
    "Calculate RMSD between reference structure and generated structure over the defined motif regions"

    # Import information about generated motifs
    motif_df = pd.read_csv(fpath+'/motif_df.csv', index_col=0)
    #print(motif_df)

    sub_motif_lens = [ref_motif_ends[j] - ref_motif_starts[j] for j in range(len(ref_motif_starts))]
    rmsds = []
    for i in range(num_structures): # This needs to be in numerical order to match new_starts file
        ref = mda.Universe(fpath+'/pdb/'+reference_PDB)
        u = mda.Universe(fpath+'/pdb/SEQUENCE_' + str(i) + '.pdb')

        ref_selection = 'name CA and resnum '
        u_selection = 'name CA and resnum '

        new_motif_starts = literal_eval(motif_df['start_idxs'].iloc[i])
        new_motif_ends = literal_eval(motif_df['end_idxs'].iloc[i])
        #print(new_motif_starts)
        print("SEQUENCE", i)
        #print("sub motif_lens", sub_motif_lens)

        for j in range(len(ref_motif_starts)):
            #print(ref_motif_starts[j]+1)
            #print(ref_motif_ends[j]+1)
            ref_selection += str(ref_motif_starts[j]+1) + ':' + str(ref_motif_ends[j]+1) + ' ' # +1 (PDB indexed at 1)
            u_selection += str(new_motif_starts[j]) + ':' + str(new_motif_ends[j]) + ' '

        # # Uncomment to make sure that selections are mapped correctly
        print("ref selection", ref_selection)
        print("len", len(ref.select_atoms(ref_selection).positions))
        print(ref.select_atoms(ref_selection).resnames)
        #print(ref.select_atoms(ref_selection)[25:31])
        print("u selection", u_selection)
        print("len", len(u.select_atoms(u_selection).positions))
        print(u.select_atoms(u_selection).resnames)
        #print(u.select_atoms(u_selection)[25:31])
        #import pdb; pdb.set_trace()

        rmsd = rms.rmsd(u.select_atoms(u_selection).positions,
                        # coordinates to align
                        ref.select_atoms(ref_selection).positions,
                        # reference coordinates
                        center=True,  # subtract the center of geometry
                        superposition=True)  # superimpose coordinates
        rmsds.append(rmsd)

    motif_df['rmsd'] = rmsds
    return motif_df

def get_confidence_score(fpath, num_structures, motif_df):
    "Get confidence score from PDB files (stored in beta)"
    scores = []
    sampled_scores = []
    fixed_scores = []

    for i in range(num_structures):
        new_motif_starts = literal_eval(motif_df['start_idxs'].iloc[i])[0]
        new_motif_ends = literal_eval(motif_df['end_idxs'].iloc[i])[-1]
        f = fpath + '/pdb/SEQUENCE_'+str(i)+'.pdb'
        # Get pdb file number
        p = PDBParser()
        structure = p.get_structure("PDB", f)
        scores_list = []
        sampled_list = []
        fixed_list = []
        for i, res in enumerate(structure.get_residues()):
            for atom in res:
                scores_list.append(atom.bfactor)
                if i < new_motif_starts:
                    sampled_list.append(atom.bfactor)
                elif i>= new_motif_starts and i<=new_motif_ends:
                    fixed_list.append(atom.bfactor)
                else:
                    sampled_list.append(atom.bfactor)
        #print(len(scores_list), len(sampled_list), len(fixed_list))
        scores.append(np.mean(scores_list))
        sampled_scores.append(np.mean(sampled_list))
        fixed_scores.append(np.mean(fixed_list))
    return scores, sampled_scores, fixed_scores

def plot_rmsd(pdb, motif_df):
    # Sort and bin, then plot
    # bins = np.arange(args.scaffold_min, args.scaffold_max + 25, 25)
    # motif_df['binned'] = pd.cut(motif_df['scaffold_lengths'], bins)

    # fig, ax = plt.subplots(figsize=(5, 3))
    fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.5))
    # sns.swarmplot(data=motif_df, x="binned", y="rmsd", ax=ax)
    ax[0].scatter(motif_df['scaffold_lengths'], motif_df['rmsd'], edgecolors='grey', c='#D0D0D0')
    ax[0].set_xlabel('Scaffold Lengths')
    ax[0].set_ylabel(r'Motif RMSD ($\AA$)')
    ax[1].scatter(motif_df['scores'], motif_df['rmsd'], edgecolors='grey', c='#D0D0D0')
    ax[1].set_xlabel('pLDDT entire sequence')
    ax[1].set_ylabel(r'Motif RMSD ($\AA$)')
    ax[2].scatter(motif_df['scores_fixed'], motif_df['rmsd'], edgecolors='grey', c='#527d99')
    ax[2].set_xlabel('pLDDT fixed region')
    ax[2].set_ylabel(r'Motif RMSD ($\AA$)')
    ax[0].axhline(y=1, c='k', ls='--', lw=0.75)
    ax[1].axhline(y=1, c='k', ls='--', lw=0.75)
    ax[2].axhline(y=1, c='k', ls='--', lw=0.75)
    plt.title("  ")
    # plt.xlabel('Scaffold Lengths')
    ax[1].set_xlim(0, 100)
    ax[2].set_xlim(0, 100)
    # ax.set_yticks([0,2,4,6,8])
    # plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(os.path.join('plots/' + pdb + '.png'))

if __name__ == '__main__':
    main()
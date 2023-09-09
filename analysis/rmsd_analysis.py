import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
import pandas as pd
import os
from evodiff.plot import plot_conditional_tmscores, plot_conditional_rmsd, plot_conditional_sim
import argparse
import difflib
from ast import literal_eval
from Bio.PDB import PDBParser, Selection
import esm.inverse_folding
import pathlib

# Get RMSD between original motif and generated motif

def main():
    # set seeds
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='oa_dm_640M',
                        help='Choice of: carp_38M carp_640M esm1b_650M \
                                  oa_dm_38M oa_dm_640M lr_ar_38M lr_ar_640M')
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
    parser.add_argument('--amlt', action='store_true')
    parser.add_argument('--random-baseline', action='store_true')
    args = parser.parse_args()

    if not args.amlt:
        home = str(pathlib.Path.home())
        if not args.random_baseline:
            home += '/Desktop/DMs/' + args.model_type + '/' + args.pdb
        else:
            home += '/Desktop/DMs/random-baseline/' + args.pdb
    else:
        home = os.getenv('AMLT_OUTPUT_DIR', '/tmp') + '/'

    if not os.path.exists(home+'plots/'):
        os.mkdir(home+'plots/')

    args.start_idxs.sort()
    args.end_idxs.sort()
    print(args.start_idxs, args.end_idxs)

    ref_pdb = args.pdb + '_reres.pdb'
    # Iterate over all generated files and calc rmsd
    motif_df = calc_rmsd(args.num_seqs, ref_pdb, fpath=home, ref_motif_starts=args.start_idxs,
                         ref_motif_ends=args.end_idxs)
    ci_scores, ci_sampled, ci_fixed = get_confidence_score(home, args.num_seqs,  motif_df)
    motif_df['scores'] = ci_scores
    motif_df['scores_sampled'] = ci_sampled
    motif_df['scores_fixed'] = ci_fixed
    motif_df_sorted = motif_df.sort_values('scores_fixed', ascending=False)
    print(motif_df_sorted[['seqs','scores','scores_sampled','scores_fixed','rmsd']])
    candidates = motif_df_sorted[(motif_df_sorted['rmsd'] <= 1) & (motif_df_sorted['scores'] >= 70)]
    print(candidates[['seqs','scores','scores_sampled','scores_fixed','rmsd']])
    print("Success:", len(candidates))
    with open(home + '/successes.csv', 'w') as f:
        f.write(str(len(candidates)) + " of " + str(args.num_seqs) + " total")
    f.close()
    motif_df.to_csv(home + '/motif_df_rmsd.csv', index=True)

    # Eval TM scores
    tm = pd.read_csv(home + '/pdb/tmscores.txt', names=['tmscore'])
    plot_conditional_tmscores(tm, ['grey'], legend=False, save_path=home+'plots/'+args.pdb)

    # Plot rmsd vs plddt
    plot_conditional_rmsd(args.pdb, motif_df, out_path=home+'plots/')

    # percent similarity in fixed region
    chain_ids=[args.chain]
    structure = esm.inverse_folding.util.load_structure(home+'/pdb/'+ref_pdb, chain_ids)
    coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
    sequence = native_seqs[chain_ids[0]]
    print("NATIVE SEQ", sequence)
    original_fixed = sequence[args.start_idxs[0]:args.end_idxs[-1]]
    sim = []
    for i in range(len(motif_df)):
        new_motif_starts = literal_eval(motif_df['start_idxs'].iloc[i])[0]
        new_motif_ends = literal_eval(motif_df['end_idxs'].iloc[i])[-1]
        gen_sequence = motif_df['seqs'].iloc[i][2:-2]
        gen_fixed = gen_sequence[new_motif_starts:new_motif_ends]
        sim.append(calc_sim(original_fixed, gen_fixed))
    # Write all scores to file
    with open(os.path.join(home + '/pdb/sim.txt'), 'w') as f:
        [f.write(str(s) + '\n') for s in sim]
    f.close()
    plot_conditional_sim(sim, out_path=home+'plots/')

def calc_sim(seq1, seq2):
    sm=difflib.SequenceMatcher(None,seq1,seq2)
    sim = sm.ratio()*100
    return sim

def calc_rmsd(num_structures, reference_PDB, fpath='conda/gen/6exz', ref_motif_starts=[30], ref_motif_ends=[44]):
    "Calculate RMSD between reference structure and generated structure over the defined motif regions"

    motif_df = pd.read_csv(fpath+'/motif_df.csv', index_col=0, nrows=num_structures)
    rmsds = []
    for i in range(num_structures): # This needs to be in numerical order to match new_starts file
        ref = mda.Universe(fpath+'/pdb/'+reference_PDB)
        u = mda.Universe(fpath+'/pdb/SEQUENCE_' + str(i) + '.pdb')

        ref_selection = 'name CA and resnum '
        u_selection = 'name CA and resnum '

        new_motif_starts = literal_eval(motif_df['start_idxs'].iloc[i])
        new_motif_ends = literal_eval(motif_df['end_idxs'].iloc[i])

        for j in range(len(ref_motif_starts)):
            ref_selection += str(ref_motif_starts[j]+1) + ':' + str(ref_motif_ends[j]+1) + ' ' # +1 (PDB indexed at 1)
            u_selection += str(new_motif_starts[j]) + ':' + str(new_motif_ends[j]) + ' '
        print("U SELECTION", u_selection)
        print("SEQUENCE", i)
        print("ref", ref.select_atoms(ref_selection).resnames)
        print("gen", u.select_atoms(u_selection).resnames)
        # This asserts that the motif sequences are the same - if you get this error something about your indices are incorrect - check chain/numbering
        assert len(ref.select_atoms(ref_selection).resnames) == len(u.select_atoms(u_selection).resnames), "Motif \
                                                                        lengths do not match, check PDB preprocessing\
                                                                        for extra residues"

        assert (ref.select_atoms(ref_selection).resnames == u.select_atoms(u_selection).resnames).all(), "Resnames for\
                                                                        motifRMSD do not match, check indexing"
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
        scores.append(np.mean(scores_list))
        sampled_scores.append(np.mean(sampled_list))
        fixed_scores.append(np.mean(fixed_list))
    return scores, sampled_scores, fixed_scores



if __name__ == '__main__':
    main()
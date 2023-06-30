import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from analysis.plot import plot_tmscore
import argparse
from ast import literal_eval

home = 'cond-gen/'

def main():
    # set seeds
    np.random.seed(0)

    parser = argparse.ArgumentParser()
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
    parser.add_argument('--scaffold-min', type=int, default=1,
                        help="Min scaffold len ")
    parser.add_argument('--scaffold-max', type=int, default=30,
                        help="Max scaffold len, will randomly choose a value between min/max")
    args = parser.parse_args()

    args.start_idxs.sort()
    args.end_idxs.sort()

    ref_pdb = args.pdb + '_reres.pdb'
    # Iterate over all generated files and calc rmsd
    motif_df = calc_rmsd(args.num_seqs, ref_pdb, fpath=home + args.pdb, ref_motif_starts=args.start_idxs,
                         ref_motif_ends=args.end_idxs)
    ci_score = get_confidence_score(home + args.pdb, args.num_seqs)
    motif_df['scores'] = ci_score
    motif_df = motif_df.sort_values('scores', ascending=False)
    print(motif_df[motif_df['rmsd'] <= 5.0])
    motif_df.to_csv(home + args.pdb + '/motif_df_rmsd.csv', index=True)

    # Uncomment if you also want to look at TM scores (need to run first)
    # tm = pd.read_csv(home+run + '/tmscores.txt', names=['tmscore'])
    # plot_tmscore(tm, ['b'], legend=False, save_file=run+'_tmscore')

    # Sort and bin, then plot
    bins = np.arange(args.scaffold_min, args.scaffold_max + 10, 10)
    motif_df['binned'] = pd.cut(motif_df['scaffold_lengths'], bins)

    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.swarmplot(data=motif_df, x="binned", y="rmsd", ax=ax)
    ax.axhline(y=1, c='k', ls='--', lw=0.75)
    plt.title(args.pdb)
    plt.xlabel('Scaffold Lengths')
    plt.ylabel('RMSD (A)')
    plt.ylim(0, 20)
    # ax.set_yticks([0,2,4,6,8])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(os.path.join('plots/' + args.pdb + '.png'))



def calc_rmsd(num_structures, reference_PDB, fpath='conda/gen/6exz', ref_motif_starts=[30], ref_motif_ends=[44]):
    "Calculate RMSD between reference structure and generated structure over the defined motif regions"

    # Import information about generated motifs
    motif_df = pd.read_csv(fpath+'/motif_df.csv', index_col=0)
    print(motif_df)

    sub_motif_lens = [ref_motif_ends[j] - ref_motif_starts[j] for j in range(len(ref_motif_starts))]
    rmsds = []
    for i in range(num_structures): # This needs to be in numerical order to match new_starts file
        ref = mda.Universe(fpath+'/pdb/'+reference_PDB)  # open AdK (PDB ID: 4AKE)
        u = mda.Universe(fpath+'/pdb/SEQUENCE_' + str(i) + '.pdb')

        ref_selection = 'name CA and resnum '
        u_selection = 'name CA and resnum '

        new_motif_starts = literal_eval(motif_df['start_idxs'].iloc[i])
        new_motif_ends = literal_eval(motif_df['end_idxs'].iloc[i])
        print("SEQUENCE", i)
        #print("sub motif_lens", sub_motif_lens)

        for j in range(len(ref_motif_starts)):
            ref_selection += str(ref_motif_starts[j]+1) + ':' + str(ref_motif_ends[j]+1) + ' ' # +1 (PDB indexed at 1)
            u_selection += str(new_motif_starts[j]) + ':' + str(new_motif_ends[j]) + ' '

        # Uncomment to make sure that selections are mapped correctly
        print("ref selection", ref_selection)
        # print("len", len(ref.select_atoms(ref_selection).positions))
        print(ref.select_atoms(ref_selection)[10:])
        print("u selection", u_selection)
        # print("len", len(u.select_atoms(u_selection).positions))
        print(u.select_atoms(u_selection)[10:])
        # import pdb; pdb.set_trace()

        rmsd = rms.rmsd(u.select_atoms(u_selection).positions,
                        # coordinates to align
                        ref.select_atoms(ref_selection).positions,
                        # reference coordinates
                        center=True,  # subtract the center of geometry
                        superposition=True)  # superimpose coordinates
        rmsds.append(rmsd)

    motif_df['rmsd'] = rmsds
    return motif_df

def get_confidence_score(fpath, num_structures):
    "Get confidence score from PDB files (stored in beta)"
    scores = []
    for i in range(num_structures):
        f = fpath + '/pdb/SEQUENCE_'+str(i)+'.pdb'
        # Get pdb file number
        df = pd.read_csv(f, delim_whitespace=True, header=None, usecols=[5,10], names=['residue','score'])
        df = df.dropna() # ignore empty rows
        if df.empty: # reading in PDBs can be finnicky if spacing is not correct
            print("confidence empty", f)
        else:
            if "C" in str(df.score):
                print(df[df.isin(["C"])])
                print("confidence", f)
                print(df.score.mean())
            else:
                key = int(df.iloc[-1]['residue']+1)
                scores.append(df.score.mean())
    return scores

if __name__ == '__main__':
    main()
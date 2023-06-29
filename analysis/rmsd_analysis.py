import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from analysis.plot import plot_tmscore
from ast import literal_eval

home = 'cond-gen/'

run = '5trv'
ref_motif_starts =  [34,105]
ref_motif_ends = [58,118]
scaffold_min = 10
scaffold_max = 50
#
# run = '6exz'
# ref_motif_start =
# ref_motif_end =
# scaffold_min = 30
# scaffold_max = 100

ref_pdb= run+'_renumbered.pdb'


def calc_rmsd(num_structures, reference_PDB=ref_pdb, fpath='conda/gen/6exz', ref_motif_starts=[30], ref_motif_ends=[44]):

    #motif_len = (max(ref_motif_ends) + 1) - min(ref_motif_starts)
    motif_df = pd.read_csv(fpath+'/motif_df.csv', index_col=0)
    print(motif_df)

    rmsds = []
    for i in range(num_structures): # This needs to be in numerical order to match new_starts file
        ref = mda.Universe(fpath+'/pdb/'+reference_PDB)  # open AdK (PDB ID: 4AKE)
        u = mda.Universe(fpath+'/pdb/SEQUENCE_' + str(i) + '.pdb')
        u_start = motif_df['start_idxs'].iloc[i]  # motif start resiude
        #u_end = u_start + (motif_len - 1)
        # print(u_start, u_end)
        # print(motif_df['seqs'].iloc[i])
        # print(motif_len)
        # print("MAKE SURE THIS ANALYSIS INCLUDES", u_end)
        # print("structure", u.select_atoms('name CA and resnum ' + str(u_start) + ':' + str(u_end)))
        # print("MAKE SURE THIS ANALYSIS INCLUDES", ref_motif_end)
        # print("reference", ref.select_atoms('name CA and resnum ' + str(ref_motif_start) + ':' + str(ref_motif_end)))

        ref_selection = 'backbone and resnum '
        u_selection = 'backbone and resnum '
        sub_motif_lens = [ref_motif_ends[j] - ref_motif_starts[j] for j in range(len(ref_motif_starts))]
        spacers = literal_eval(motif_df['spacers'].iloc[i])
        print("SEQUENCE", i)
        print("sub motif_lens", sub_motif_lens)
        print("new motif start", u_start)
        print("spacers", spacers)
        u_start_new = u_start
        for j in range(len(ref_motif_starts)):
            print("in loop", j)
            print(sub_motif_lens[j])
            #print(spacers[j])
            if j <= 0:
                u_selection += str(u_start) + ':' + str(u_start+sub_motif_lens[j])
            if j > 0:
                ref_selection += ' '
                u_selection += ' '
                u_selection += str(u_start_new) + ':' + str(u_start_new+sub_motif_lens[j])
            ref_selection += str(ref_motif_starts[j]) + ':' + str(ref_motif_ends[j])
            if j < len(ref_motif_starts)-1:
                u_start_new += sub_motif_lens[j] + spacers[j]

        print(ref_selection)
        print("len", len(ref.select_atoms(ref_selection).positions))
        print(u_selection)
        print("len", len(u.select_atoms(u_selection).positions))
        # # import pdb; pdb.set_trace()

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
                #print(key)
                scores.append(df.score.mean())
    return scores

# Iterate over all generated files and calc rmsd
num_structures = 100
motif_df = calc_rmsd(num_structures, reference_PDB=ref_pdb, fpath=home+run, ref_motif_starts=ref_motif_starts,
                     ref_motif_ends=ref_motif_ends)
ci_score = get_confidence_score(home+run, num_structures)
motif_df['scores'] = ci_score
motif_df = motif_df.sort_values('scores', ascending=False)
print(motif_df[motif_df['rmsd'] <= 5.0])
motif_df.to_csv(home+run + '/motif_df_rmsd.csv', index=True)

# tm = pd.read_csv(home+run + '/tmscores.txt', names=['tmscore'])
# plot_tmscore(tm, ['b'], legend=False, save_file=run+'_tmscore')

# Sort and bin, then plot
bins = np.arange(scaffold_min, scaffold_max+10, 10)
motif_df['binned'] = pd.cut(motif_df['scaffold_lengths'], bins)

fig, ax = plt.subplots(figsize=(5, 2.5))
sns.boxplot(data=motif_df, x="binned", y="rmsd", ax=ax)
#plt.plot(plot_centers, plot_values, c='b', marker='o')

ax.axhline(y=1, c='k', ls='--', lw=0.75)
plt.title(run)
plt.xlabel('Scaffold Lengths')
plt.ylabel('RMSD (A)')
plt.ylim(0,20)
#ax.set_yticks([0,2,4,6,8])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
fig.savefig(os.path.join('plots/'+run+'.png'))
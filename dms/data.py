# import pandas as pd
# from dms.utils import parse_fasta
# from torch.utils.data import Dataset
#
# # TODO: replace w/ Kevins sequence_models.datasets.FlatDataset
# class UNIREF50(Dataset):
#     """
#     Dataset stores samples and labels
#     Has been preprocessed w/ dms.utils.read_fasta first to create an SEQ, INFO, INDEX file that contains
#     sequences, headers, and indices from original dataset
#     """
#     def __init__(self, index_file, seq_file):
#         self.index_file = index_file
#         self.seq_file = seq_file
#
#     def __len__(self):
#         index = pd.read_csv(self.index_file, header=None, names=['index'], index_col=['index'])
#         return len(index)
#
#     def __getitem__(self, idx):
#         sequence = parse_fasta(self.seq_file, idx)
#         return(sequence)

def loadMatrix(path):
    """
    Reads a Blosum matrix from file.
    File in a format like:
        https://www.ncbi.nlm.nih.gov/IEB/ToolBox/C_DOC/lxr/source/data/BLOSUM62
    Input:
        path: str, path to a file.
    Returns:
        blosumDict: Dictionary, The blosum dict
    """

    with open(path, "r") as f:
        content = f.readlines()

    blosumDict = {}

    header = True
    for line in content:
        line = line.strip()

        # Skip comments starting with #
        if line.startswith("#"):
            continue

        linelist = line.split()

        # Extract labels only once
        if header:
            labelslist = linelist
            header = False

            # Check if all AA are covered
            #if not len(labelslist) == 25:
            #    print("Blosum matrix may not cover all amino-acids")
            continue

        if not len(linelist) == len(labelslist) + 1:
            # Check if line has as may entries as labels
            raise EOFError("Blosum file is missing values.")

        # Add Line/Label combination to dict
        for index, lab in enumerate(labelslist, start=1):
            blosumDict[f"{linelist[0]}{lab}"] = float(linelist[index])

    # Check quadratic
    if not len(blosumDict) == len(labelslist) ** 2:
        raise EOFError("Blosum file is not quadratic.")
    return blosumDict

import pandas as pd
from dms.utils import parse_fasta
from torch.utils.data import Dataset

# TODO: replace w/ Kevins sequence_models.datasets.FlatDataset
class UNIREF50(Dataset):
    """
    Dataset stores samples and labels
    Has been preprocessed w/ dms.utils.read_fasta first to create an SEQ, INFO, INDEX file that contains
    sequences, headers, and indices from original dataset
    """
    def __init__(self, index_file, seq_file):
        self.index_file = index_file
        self.seq_file = seq_file

    def __len__(self):
        index = pd.read_csv(self.index_file, header=None, names=['index'], index_col=['index'])
        return len(index)

    def __getitem__(self, idx):
        sequence = parse_fasta(self.seq_file, idx)
        return(sequence)


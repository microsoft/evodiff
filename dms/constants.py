from sequence_models.constants import ALL_AAS, SPECIALS

ROUND=5 #rounding q matrix

PAD = '!'

PROTEIN_ALPHABET = ALL_AAS + SPECIALS + PAD

BLOSUM62_ALPHABET = 'ARNDCQEGHILKMFPSTWYVBZX' # In order of BLOSUM indices for matrix creation

### CONSTANTS RELEVANT TO PROTEIN VOCAB ###
# CAN_AAS = 'ACDEFGHIKLMNPQRSTVWY'
# # single AA letter codes
#
# AMB_AAS = 'BZX'
# # B: aspartic acid (D) or asparagine (N)
# # Z: glu acid (E) or glutamine (Q)
# # X: any
#
# OTHER_AAS = 'JOU' # AND THESE
# # J: leucine (L) or isoleucine (I)
# # O: Pyrrolysine
# # U: Selenocystine
#
# ALL_AAS = CAN_AAS + AMB_AAS + OTHER_AAS
#
# PAD = '!'
#
# STOP = '*'
# GAP = '-'
# MASK = '#'  # Useful for masked language model training
# START = '@'
# SPECIALS = STOP + GAP + MASK + START
#
# PROTEIN_ALPHABET = ALL_AAS + SPECIALS + PAD
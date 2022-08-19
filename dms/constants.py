from sequence_models.constants import MASK, SPECIALS

BLOSUM62_AAS = 'ARNDCQEGHILKMFPSTWYVBZX' # In order of BLOSUM indices for matrix creation
OTHER_AAS = 'JOU'
PAD = '!'

ALL_AAS = BLOSUM62_AAS + OTHER_AAS

PROTEIN_ALPHABET = ALL_AAS + PAD + MASK #SPECIALS

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
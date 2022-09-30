from sequence_models.constants import AAINDEX_ALPHABET, AMB_AAS, OTHER_AAS, GAP, MASK, SPECIALS, STOP, MASK, START, ALL_AAS

# ORDER is important for BLOSUM indexing
MSA_PAD='!'

# FOR BLOSUM MATRIX CREATION ONLY (In order of BLOSUM indices for mapping onto MSA_ALPHABET)
BLOSUM_AAS = AAINDEX_ALPHABET + AMB_AAS
BLOSUM_ALPHABET = BLOSUM_AAS + OTHER_AAS + GAP + MSA_PAD + STOP + MASK + START
BLOSUM_EXTRAS = 'JOU-'

MSA_AAS = ALL_AAS + GAP
MSA_ALPHABET = ALL_AAS + GAP + STOP + MASK + START + MSA_PAD

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
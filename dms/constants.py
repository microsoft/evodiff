from sequence_models.constants import AAINDEX_ALPHABET, AMB_AAS, OTHER_AAS, GAP, MASK, SPECIALS, STOP, MASK, START

# ORDER is important for BLOSUM indexing
MSA_PAD='!'

BLOSUM62_AAS = AAINDEX_ALPHABET + AMB_AAS + OTHER_AAS # In order of BLOSUM indices for matrix creation

ALL_AAS = BLOSUM62_AAS
MSA_ALL_AAS =  BLOSUM62_AAS + GAP

PROTEIN_ALPHABET = ALL_AAS + MSA_PAD + MASK #SPECIALS
MSA_ALPHABET_NEW = MSA_ALL_AAS + MSA_PAD + STOP + MASK + START

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
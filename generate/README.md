# All conditional generation runs 
# Start/end indexes are adjusted for the protein sequence indexed at 0 
# If the domain of interest 

# PDB CODE: 5YUI
# Domains: 94-96 (HFH), 114-116 (LHL), 101-104(SEHT) (8 residues)
# Scaffold length range: 60-105 
python generate/conditional_generation.py --cond-task scaffold --pdb 5YUI --start-idxs 90 --end-idxs 92 --start-idxs 114 
    --end-idxs 116 --start-idxs 101 --end-idxs 104  --num-seqs 100 --scaffold-min 60 --scaffold-max 105

# PDB CODE: 1QJG
# Domains: Y14, N36, D99 (3 residues)
# Scaffold length range: 50-100
python generate/conditional_generation.py --cond-task scaffold --pdb 1qjg --start-idxs 13 --end-idxs 13 --start-idxs 37 
    --end-idxs 37 --start-idxs 98 --end-idxs 98 --num-seqs 100 --scaffold-min 50 --scaffold-max 100

# PDB CODE: 1PRW
# Domains: 16-35, 52-71
# Scaffold length range: 15-65
python generate/conditional_generation.py --cond-task scaffold --pdb 1PRW --start-idxs 14 --end-idxs 33 --start-idxs 50 
    --end-idxs 69 --num-seqs 10 --scaffold-min 15 --scaffold-max 65

# PDB CODE: 1BCF 
# Domains: 18-25, 47-54, 94-97, 123-130
python generate/conditional_generation.py --cond-task scaffold --pdb 1bcf --start-idxs 17 --end-idxs 26 --start-idxs 46
    --end-idxs 53 --start-idxs 93 --end-idxs 96 --start-idxs 122 --end-idxs 129 --scaffold-min 50 --scaffold-max 100 
    --num-seqs 100

# PDB CODE: 5TPN (Site V)
# Domain: 163-181
python generate/conditional_generation.py --cond-task scaffold --pdb 5tpn --start-idx 108 --end-idx 126
    --scaffold-min 50 -scaffold-max 100 --num-seqs 100

# PDB CODE: 5TPN (Site II)
# Domain: 254-277
python generate/conditional_generation.py --cond-task scaffold --pdb 5tpn --start-idx 200 --end-idx 222
    --scaffold-min 50 -scaffold-max 100 --num-seqs 100

# PDB CODE: 5TPN (Site 0)
# Domain: 63-69, 196-212
python generate/conditional_generation.py --cond-task scaffold --pdb 5tpn --start-idx 36 --end-idx 41 --start-idx 141 
    --end-idx 157 --scaffold-min 50 --scaffold-max 100 --num-seqs 100 

# PDB CODE: 5TPN (Site IV)
# Domain: 422-436
python generate/conditional_generation.py --cond-task scaffold --pdb 5tpn --start-idx 367 --end-idx 381
    --scaffold-min 50 -scaffold-max 100 --num-seqs 100

# PDB CODE: 5IUS 
# Domains: 63-82, 119-140
generate/conditional_generation.py --cond-task scaffold --pdb 5ius --start-idx 34 --end-idx 53 --start-idx 88 
    --end-idx 109 --scaffold-min 50 --scaffold-max 100 --num-seqs 100

# PBD CODE: 1YCR
# Domains: 19-27
python generate/conditional_generation.py --cond-task scaffold --pdb 1ycr --start-idx 2 --end-idx 10 --scaffold-min 50 
    --scaffold-max 100 --num-seqs 100 --chain B 

# PDB CODE: 5WN9
# Domains 17-191


# PDB CODE: 5IUS
# Domains 63-82, 119-140

# PDB CODE: 1YCR
# Domains 19-27

# PDB CODE: 7P19
# Domain: 24-43
python generate/conditional_generation.py --cond-task scaffold --pdb 7p19 --start-idx 5 --end-idx 23 --scaffold-min 50 
    --scaffold-max 100 --num-seqs 100

# PDB CODE: 2KL8 
# Domain: 1-8, 27-78

# Domain: 8-27

# PDB CODE: 7MRX 
# Domain: 25-46

# PDB CODE: 5TRV
# Domain 45-69

# PDB CODE: 6EXZ
# Domain 28-42

# PDB CODE: 6E6R
# Domain 23-35

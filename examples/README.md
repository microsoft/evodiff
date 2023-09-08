# IDR generation 
From sequence model 
```
python generate/conditional_generation.py --model-type oa_ar_640M --cond-task idr 
```
From MSA model 
```
TODO
```

# Scaffolding Task
Benchmark conditional generation runs from Table S9 https://www.nature.com/articles/s41586-023-06415-8

Start/end indexes are adjusted for the protein sequence extracted from PDB files

Corresponding motif domains held frozen for each scaffolding task are listed for reference

All motifs were generated in a 50-100 residue scaffold

### PDB CODE: 1PRW
Domains: 16-35 (FSLFDKDGDGTITTKELGTV), 52-71 (INEVDADGNGTIDFPEFLTM)
```
python generate/conditional_generation.py --model-type oa_ar_640M --cond-task scaffold --pdb 1prw --start-idxs 15 --end-idxs 34 --start-idxs 51 --end-idxs 70 --num-seqs 100 --scaffold-min 50 --scaffold-max 100
```
Equivalent code for generating from MSA:
```
python generate/conditional_generation_msa.py --model-type msa_oa_ar_maxsub --cond-task scaffold --pdb 1prw --start-idxs 15 --end-idxs 34 --start-idxs 51 --end-idxs 70 --num-seqs 1 --query-only
```
### PDB CODE: 1BCF 

Domains:  18-25 (ELVAINQY), 47-54 (ESIDEMKH), 91-99 (LALELDGAK), 123-130 (ILRDEEGH)
```
python generate/conditional_generation.py --cond-task scaffold --pdb 1bcf --start-idxs 17 --end-idxs 24 --start-idxs 46 --end-idxs 53 --start-idxs 90 --end-idxs 98 --start-idxs 122 --end-idxs 129 --scaffold-min 50 --scaffold-max 100 --num-seqs 100
```

### PDB CODE: 5TPN (Site V)
Domain: 163-181 (EVNKIKSALLSTNKAVVSL)
```
python generate/conditional_generation.py --cond-task scaffold --pdb 5tpn --start-idx 108 --end-idx 126 --scaffold-min 50 -scaffold-max 100 --num-seqs 100
```

### PDB CODE: 3IXT
Domain: 254-277 (NSELLSLINDMPITNDQKKLMSNN)
```
python generate/conditional_generation.py --cond-task scaffold --pdb 3ixt --start-idx 0 --end-idx 23 --scaffold-min 50 --scaffold-max 100 --num-seqs 100 --chain P
```
analysis indices to account for chain:
```
python analysis/rmsd_analysis.py --pdb 3ixt --start-idx 424 --end-idx 447 --scaffold-min 50 --scaffold-max 100 --num-seqs 1 --chain P
```

[//]: # (### PDB CODE: 5TPN &#40;Site 0&#41; -> &#40;4JHW Baker reference&#41;)

[//]: # (Domain: 63-69 &#40;NIKKIKC&#41;, 196-212 &#40;KNYIDKQLLPIVNKQSC&#41;)

[//]: # (```)

[//]: # (python generate/conditional_generation.py --cond-task scaffold --pdb 5tpn --start-idx 36 --end-idx 42 --start-idx 141 --end-idx 157 --scaffold-min 50 --scaffold-max 100 --num-seqs 100 )

[//]: # (```)

# PDB CODE: 4JHW
Domain: Chain F 63-69(NIKENKC), 196-212(KQSCSISNIETVIEFQ), 
```
python generate/conditional_generation.py --cond-task scaffold --pdb 4jhw --start-idx 37 --end-idx 43 --start-idx 144 --end-idx 159 --scaffold-min 50 --scaffold-max 100 --num-seqs 1 --chain F
```
analysis indices to account for chain :
```
python analysis/rmsd_analysis.py --pdb 4jhw --start-idx 474 --end-idx 480 --start-idx 581 --end-idx 596 --scaffold-min 50 --scaffold-max 100 -
-num-seqs 1 --chain F
```

[//]: # (### PDB CODE: 5TPN &#40;Site IV&#41; -> &#40;4ZYP Baker reference&#41;)

[//]: # (Domain: 422-436 &#40;CTASNKNRGIIKTFS&#41;)

[//]: # (```)

[//]: # (python generate/conditional_generation.py --cond-task scaffold --pdb 5tpn --start-idx 367 --end-idx 381 --scaffold-min 50 -scaffold-max 100 --num-seqs 100)

[//]: # (```)
### PDB CODE: 4ZYP 
Domain 422-436(CTASNKNRGIIKTFS)
```
python generate/conditional_generation.py --cond-task scaffold --pdb 4zyp --start-idx 357 --end-idx 371 --scaffold-min 50 --scaffold-max 100 --num-seqs 100 --chain A
```

### PDB CODE: 5WN9
Domains 170-189 (FVPCSICSNNPTCWAICKRI)
```
python generate/conditional_generation.py --cond-task scaffold --pdb 5wn9 --start-idx 1 --end-idx 20 --scaffold-min 50 -scaffold-max 100 --num-seqs 100
```
analysis needs diff indices b/c postprocessed-pdb file retains chain indexing  
```
python analysis/rmsd_analysis.py --pdb 5wn9 --start-idxs 237 --end-idxs 257 --num-seqs 1 --scaffold-min 50 
    --scaffold-max 100
```
### PDB CODE: 5IUS 
Domains: 63-82 (FHVVWHRESPSGQTDTLAAF), 119-140 (GTYVCGVISLAPKIQIKESLRA)
```
python generate/conditional_generation.py --cond-task scaffold --pdb 5ius --start-idx 34 --end-idx 53 --start-idx 88 --end-idx 109 --scaffold-min 50 --scaffold-max 100 --num-seqs 100
```

### PDB CODE: 5YUI
Domains: 93-97 (FHFHW), 118-120 (LHL), 198-200 (TTP)
```
python generate/conditional_generation.py --model-type oa_ar_640M --cond-task scaffold --pdb 5yui --start-idxs 89 --end-idxs 93 --start-idxs 114 --end-idxs 116 --start-idxs 194 --end-idxs 196  --num-seqs 100 --scaffold-min 50 --scaffold-max 100
```

[//]: # (### PDB CODE: 7P19 -> &#40;6VW1&#41;)

[//]: # (Domain: 24-42 &#40;QAKTFLDKFNHEAEDLFYQ&#41;, 64-82 &#40;NAGDKWSAFLKEQSTLAQM&#41; )

[//]: # ()
[//]: # (WARNING: 7P19 has a residue "BLYS" that complicates generaton/analysis - pre-clean downloaded PDB &#40;delete BLYS residue&#41; before running)

[//]: # (```)

[//]: # (python generate/conditional_generation.py --cond-task scaffold --pdb 7p19 --start-idx 5 --end-idx 23 --start-idx 45 --end-idx 63 --scaffold-min 50 --scaffold-max 100 --num-seqs 100)

[//]: # (```

# PDB CODE: 6VW1
Domains: 24-42(QAKTFLDKFNHEAEDLFYQ), 64-82(NAGDKWSAFLKEQSTLAQM)
```
python generate/conditional_generation.py --cond-task scaffold --pdb 6vw1 --start-idx 5 --end-idx 23 --start-idx 45 --end-idx 63 --scaffold-min 50 --scaffold-max 100 --num-seqs 100)
```

### PDB CODE: 1QJG
Domains: 14(Y), 38 (N), 99 (D)
```
python generate/conditional_generation.py --cond-task scaffold --pdb 1qjg --start-idxs 13 --end-idxs 13 --start-idxs 37 --end-idxs 37 --start-idxs 98 --end-idxs 98 --num-seqs 100 --scaffold-min 50 --scaffold-max 100 --single-res-domain
```

### PBD CODE: 1YCR
Domains: 19-27 (Chain B: FSDLWKLLP)
```
python generate/conditional_generation.py --cond-task scaffold --pdb 1ycr --start-idx 2 --end-idx 10 --scaffold-min 50 --scaffold-max 100 --num-seqs 100 --chain B
```
analysis needs diff indices b/c postprocessed-pdb file retains chain indexing  
```
python generate/conditional_generation.py --cond-task scaffold --pdb 1ycr --start-idx 87 --end-idx 95 --scaffold-min 50 --scaffold-max 100 --num-seqs 100 --chain B
```

### PDB CODE: 2KL8 
Task 1 - Domain: 1-7 (MEMDIRF), 28-79 (KFAGTVTYTLDGNDLEIRITGVPEQVRKELAKEAERLAKEFNITVTYTIRLE)
```
python generate/conditional_generation.py --cond-task scaffold --pdb 2kl8 --start-idx 0 --end-idx 6 --start-idx 26 --end-idx 78 --scaffold-min 50 --scaffold-max 100 --num-seqs 100 
```
Task 2 - Domain: 8-27 (RGDDLEAFEKALKEMIRQAR)
```
python generate/conditional_generation.py --cond-task scaffold --pdb 2kl8 --start-idx 7 --end-idx 26 --scaffold-min 50 --scaffold-max 100 --num-seqs 100 
```

### PDB CODE: 7MRX 
Domain: 25-46 (ALPEYYGENLDALWDALTGWVE) 
```
python generate/conditional_generation.py --cond-task scaffold --pdb 7mrx --start-idx 25 --end-idx 46 --scaffold-min 50 --scaffold-max 100 --num-seqs 100 --chain B 
```
For MSAs: 
```
python generate/conditional_generation_msa.py --cond-task scaffold --pdb 7mrx --start-idx 25 --end-idx 46 --num-seqs 100 
```
analysis needs diff indices b/c postprocessed-pdb file retains chain indexing  
```
python analysis/rmsd_analysis.py --model-type msa_oa_ar_maxsub --pdb 7mrx --start-idx 133 --end-idx 154 --num-seqs 100
```

### PDB CODE: 5TRV
Domain 45-69 (EEAEKMWRKLMKFVDRVEVRRVKVD)
```
python generate/conditional_generation.py --cond-task scaffold --pdb 5trv --start-idx 45 --end-idx 69 --scaffold-min 50 --scaffold-max 100 --num-seqs 100 
```

### PDB CODE: 6E6R
Domain 23-35 (CSYEEVREATGVG)
```
python generate/conditional_generation.py --cond-task scaffold --pdb 6e6r --start-idx 22 --end-idx 34 --scaffold-min 50 --scaffold-max 100 --num-seqs 100
```

### PDB CODE: 6EXZ
Domain 28-42 (LHLETKLNAEYTFML)
```
    python generate/conditional_generation.py --cond-task scaffold --pdb 6exz --start-idx 25 --end-idx 39 --scaffold-min 50 --scaffold-max 100 --num-seqs 100
```

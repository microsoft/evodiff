#!/bin/bash

for i in {1..50}; do
  cd gen-$i

  # make fasta files
  head -n 2 valid_msas.a3m > valid_query.fasta
  head -n 2 generated_msas.a3m > gen_query.fasta
  # make msa folder for alphafold
  mkdir valid_query
  mkdir gen_query
  mkdir valid_query/msas
  mkdir gen_query/msas
  # move msas into folder
  cp valid_msas.a3m valid_query/msas/bfd_uniref_hits.a3m
  cp generated_msas.a3m gen_query/msas/bfd_uniref_hits.a3m

  cd ../

done



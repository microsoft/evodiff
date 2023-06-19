#!/bin/bash

for i in {1..50}; do
  tmscore gen-$i/gen_query/ranked_0.pdb gen-$i/valid_query/ranked_0.pdb | grep "TM-score    =" | awk -v RUN=$i '{print "gen-$RUN", $3}' >> tmscores.txt
done

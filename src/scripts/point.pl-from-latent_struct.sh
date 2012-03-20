#!/bin/bash

# Input file should be of the form used in eval_predictions.txt.gz

f=${1}
zcat ${f} | awk '$3==0{print "- dummy",$10} $3==1{print "+ dummy",$10}' | sort -nrk 3 | perl point.pl -r3

exit 0


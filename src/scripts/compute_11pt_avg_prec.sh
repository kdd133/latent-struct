#!/bin/bash
#
# Computes 11-point average precision for all the job results in a completed
# experiment.
#
# The script should be run from within the scripts directory, while providing
# an absolute path to the base experiment directory.
#
# This script expects point.pl-from-latent_struct.sh to be in the scripts
# directory as well.

if [ "$1" == "" ]; then
  echo "Usage: ./compute_11pt_avg_prec.sh <experiment directory>"
  exit 1
fi

for f in ${1}/*/*-train_predictions.txt.gz
do
  ./point.pl-from-latent_struct.sh ${f} | gzip > ${f%-*}-train_11pt_avg_prec.txt.gz
done

# Note: There may be multiple eval files, but this script will only process the
# one with id 0, i.e., eval0.
for f in ${1}/*/*-eval0_predictions.txt.gz
do
  ./point.pl-from-latent_struct.sh ${f} | gzip > ${f%-*}-eval0_11pt_avg_prec.txt.gz
done

exit 0


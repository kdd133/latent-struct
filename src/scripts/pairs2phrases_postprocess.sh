#!/bin/bash
#
# Combines phrase pairs output by Shane Bergsma's pairs2phrases.pl with
# labelled source-target pairs. This format is read by PhrasePairsReader.cpp.
#
# Example usage:
# pairs2phrases_postprocess.sh Data/Phrases/rs-en.prepared.0.58.train.phrases.3 Data/Dictionary/FVInput/rs-en.prepared.0.58.train

phrases=${1}
pairs=${2}

if [[ "$phrases" == "" || "$pairs" == "" ]]; then
  echo "Usage: pairs2phrases_postprocess.sh <phrases file> <pairs file>"
  exit 1
fi

# get rid of "--" lines
grep -v '^\-\-' $phrases > /tmp/phrases_clean

paste $pairs /tmp/phrases_clean

exit 0

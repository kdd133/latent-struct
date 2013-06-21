#!/bin/bash

f=$1

if [ "${f}" == "" ]; then
  echo "Usage: generate_ngrams.sh <words file> <n low> <n high>"
  exit 1
fi

for n in `seq $2 $3`; do
  ngram-count -write ${f}.${n}grams -text ${f} -write-order ${n} -order ${n}
  echo "wrote ${f}.${n}grams"
done

exit 0

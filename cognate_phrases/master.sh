#!/bin/bash

f=${1}

nglo=2
nghi=4
cut=0.95
k=250
#method="cut"

if [ "${f}" == "" ]; then
  echo "Usage: master.sh <cognate .train file>"
  exit 1
fi

src=${f##*/}.source
echo "python2 cognate_extract.py ${f} ${src} source"
python2 cognate_extract.py ${f} ${src} source
echo "wrote ${src}"

tgt=${f##*/}.target
echo "python2 cognate_extract.py ${f} ${tgt} target"
python2 cognate_extract.py ${f} ${tgt} target
echo "wrote ${tgt}"


echo "./generate_ngrams.sh ${src} ${nglo} ${nghi}"
./generate_ngrams.sh ${src} ${nglo} ${nghi}

echo "./generate_ngrams.sh ${tgt} ${nglo} ${nghi}"
./generate_ngrams.sh ${tgt} ${nglo} ${nghi}


if [ "${method}" == "cut" ]; then
  # Create n-gram lexicons based on highest percentage of n-grams.
  python2 extract_freq_ngrams.py -c ${cut} -o ${src}.nglex ${src}.*grams
  python2 extract_freq_ngrams.py -c ${cut} -o ${tgt}.nglex ${tgt}.*grams
else
  # Create n-gram lexicons based on k most frequent n-grams.
  python2 extract_freq_ngrams.py -k ${k} -o ${src}.nglex ${src}.*grams
  python2 extract_freq_ngrams.py -k ${k} -o ${tgt}.nglex ${tgt}.*grams
fi

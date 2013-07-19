#!/bin/bash

f=${1}
nghi=${2}

nglo=2
cut=0.95
k=250
p=0.10
#method="cut"
#method="top"

if [ "${f}" == "" ] || [ "${nghi}" == "" ]; then
  echo "Usage: master.sh <cognate .train file> <max n-gram size>"
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

ext=${nghi}glex
if [ "${method}" == "cut" ]; then
  # Create n-gram lexicons based on highest percentage of n-grams.
  python2 extract_freq_ngrams.py -c ${cut} -o ${src}.c${cut}.${ext} ${src}.*grams
  python2 extract_freq_ngrams.py -c ${cut} -o ${tgt}.c${cut}.${ext} ${tgt}.*grams
elif [ "${method}" == "top" ]; then
  # Create n-gram lexicons based on k most frequent n-grams.
  python2 extract_freq_ngrams.py -k ${k} -o ${src}.k${k}.${ext} ${src}.*grams
  python2 extract_freq_ngrams.py -k ${k} -o ${tgt}.k${k}.${ext} ${tgt}.*grams
else
  # Create n-gram lexicons that make up the given amount of probability mass.
  python2 extract_freq_ngrams.py -p ${p} -o ${src}.p${p}.${ext} ${src}.*grams
  python2 extract_freq_ngrams.py -p ${p} -o ${tgt}.p${p}.${ext} ${tgt}.*grams
fi

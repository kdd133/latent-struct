#!/bin/bash

train=${1}

if [ "$train" == "" ]; then
  echo "Usage: create_cognate_al_data.sh <train file>"
  exit 1
fi

f=${train%.*}
f=${f##*/}

grep + $train > $f.seed
grep -v + $train > $f.pool
head -n1 $f.pool >> $f.seed
sed -i '1d' $f.pool
cat $f.seed $f.pool > $f.seed+pool

./latent_struct --train=$f.seed+pool --model=StringEdit --reader=CognatePairAligner --fgen-lat=Empty --fgen-obs=BKWord --threads=8 --bias-no-normalize --add-begin-end --substring-size=3 --save-features=$f.seed+pool.alphabet

exit 0

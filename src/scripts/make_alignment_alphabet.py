#!/usr/bin/env python3

"""
Takes an FVInput file (lines of form: + source target) and outputs an alphabet
that contains a feature for each possible (source,target) character pair,
as well as begin/end markers and epsilon (insert/delete) symbols. The script
also outputs a second file containing weights that produce "default" minimum
edit distance (i.e., Levenschtein) alignments when used in latent_struct.

Note 1: If the weights are to be loaded into latent_struct for training a model
that has both w and u parameters, they must first be processed using the
make_weights_multiclass.py script (with the number of classes set to 1 in case
of a binary model), e.g.:
  make_weights_multiclass.py w_weights u_weights 1

Note2 : This script creates features and weights that represent a binary
classification model. If a multiclass model is desired, simply change the first
line of the alphabet file from '-1 0'  to '0 1 ... (k-1)' where k is the number
of class labels.
"""

import codecs
import sys

from math import log

def main():
  if len(sys.argv) != 3:
    print('Usage: %s <FVInput file> <output file>' % sys.argv[0])
    return
  
  # These must match the values in FeatureGenConstants.cpp
  OP_SEP = '>'
  EPSILON = '-'
  BEGIN_CHAR = '^'
  END_CHAR = '$'
  
  fname = sys.argv[1]
  fname_out = sys.argv[2]
  
  chars_source = set([EPSILON, BEGIN_CHAR, END_CHAR])
  chars_target = set([EPSILON, BEGIN_CHAR, END_CHAR])
  for line in codecs.open(fname, encoding='latin1').readlines():
    tokens = line.split()
    assert len(tokens) == 3
    source = tokens[1]
    target = tokens[2]
    for char in source:
      chars_source.add(char)
    for char in target:
      chars_target.add(char)

  alphabet_out = codecs.open(fname_out, encoding='latin1', mode='w')  
  alphabet_out.write('-1 0\n')
  count = 0
  for s in sorted(chars_source):
    for t in sorted(chars_target):
      if s != EPSILON or t != EPSILON:
        alphabet_out.write('%d A:%s%s%s\n' % (count, s, OP_SEP, t))
        count += 1
  alphabet_out.close()
  print('Wrote ' + fname_out)
  
  weights_out = open(fname_out + '.weights', 'w')
  i = 0
  for s in sorted(chars_source):
    for t in sorted(chars_target):
      if s == EPSILON and t == EPSILON:
        continue
      if s == t: # MATCH
        w = 1e8
      elif s == EPSILON or t == EPSILON: # INS or DEL
        w = 1
      else: # SUB
        w = 1/3
      weights_out.write('%d %g\n' % (i, log(w)))
      i += 1  
  weights_out.close()
  print('Wrote ' + fname_out + '.weights')

if __name__ == '__main__':
  main()

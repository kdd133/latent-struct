#!/usr/bin/env python3

"""
Takes an FVInput file (lines of form: + source target) and outputs an alphabet
that contains a feature for each possible (source,target) character pair,
as well as begin/end markers and epsilon (insert/delete) symbols. The script
also outputs a second file containing weights that produce "default" minimum
edit distance (i.e., Levenschtein) alignments when used in latent_struct.
"""

import codecs
import sys

from math import log

def main():
  if len(sys.argv) != 4:
    print('Usage: %s <FVInput file> <output file> <binary|multi>' % sys.argv[0])
    return
  
  # These must match the values in FeatureGenConstants.cpp
  OP_SEP = '>'
  EPSILON = '-'
  BEGIN_CHAR = '^'
  END_CHAR = '$'
  
  fname = sys.argv[1]
  fname_out = sys.argv[2]
  multiclass = sys.argv[3].startswith('multi')
  
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
  if multiclass:
    alphabet_out.write('0 1\n')
  else:
    alphabet_out.write('-1 0\n')
  count = 0
  for s in sorted(chars_source):
    for t in sorted(chars_target):
      if s != EPSILON or t != EPSILON:
        alphabet_out.write('%d A:%s%s%s\n' % (count, s, OP_SEP, t))
        count += 1
  alphabet_out.close()
  
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
      if multiclass:
        weights_out.write('%d %g\n' % (i + count, log(w)))
      i += 1  
  weights_out.close()

if __name__ == '__main__':
  main()

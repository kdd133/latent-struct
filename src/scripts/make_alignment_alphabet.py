#!/usr/bin/env python3

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
  weights_out = open(fname_out + '.weights', 'w')
  
  alphabet_out.write('-1 0\n')
  i = 0
  for s in sorted(chars_source):
    for t in sorted(chars_target):
      if s == EPSILON and t == EPSILON:
        continue
      alphabet_out.write('%d A:%s%s%s\n' % (i, s, OP_SEP, t))
      if s == t: # MATCH
        weights_out.write('%d %g\n' % (i, log(1e8)))
      elif s == EPSILON or t == EPSILON: # INS or DEL
        weights_out.write('%d %g\n' % (i, log(1)))
      else: # SUB
        weights_out.write('%d %g\n' % (i, log(1/3)))
      i += 1

  alphabet_out.close()
  weights_out.close()

if __name__ == '__main__':
  main()

#!/usr/bin/env python3

"""
Takes a weights file as input, and outputs (to stdout) the same set of weights,
but duplicated for a second class label.
"""

import sys

def main():
  if len(sys.argv) != 2:
    print('Usage: %s <weights file>' % sys.argv[0])
    return
  
  fname = sys.argv[1]
  
  weights = []
  for line in open(fname):
    print(line, end='')
    weights.append(line.split()[-1])
  
  n = len(weights)
  for (i,w) in enumerate(weights):
    print('%d %s' % (i+n, w))
  
if __name__ == '__main__':
  main()

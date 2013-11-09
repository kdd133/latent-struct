#!/usr/bin/env python3

"""
Takes two weights files as input: one for the w vector and one for the u vector.
The program outputs two files (whose filenames are the input files with ".multi"
appended) that merge the weights from the input files. That is, each weight is
duplicated k times (where k is specified by the <num classes> argument), and the
indices are sequenced as [w1 u1 w2 u2 ... wk uk], where wj is the jth copy of w.

If there are exactly two classes, the w weights for class 0 are negated.
"""

import sys

def main():
  if len(sys.argv) != 4:
    print('Usage: %s <w weights file> <u weights file> <num classes>' % sys.argv[0])
    return
  
  fname_w = sys.argv[1]
  fname_u = sys.argv[2]
  num_classes = int(sys.argv[3])
  
  weights_w = []
  for line in open(fname_w):
    weights_w.append(float(line.split()[-1]))
    
  weights_u = []
  for line in open(fname_u):
    weights_u.append(float(line.split()[-1]))
  
  nw = len(weights_w)
  nu = len(weights_u)
  n = nw + nu
  
  fname_out_w = '%s.nc%d' % (fname_w, num_classes)
  fname_out_u = '%s.nc%d' % (fname_u, num_classes)
  
  out_w = open(fname_out_w, 'w')
  out_u = open(fname_out_u, 'w')

  for k in range(num_classes):
    w_sign = (k == 0 and num_classes == 2) and -1.0 or 1.0
    for (i,w) in enumerate(weights_w):
      if w != 0:
        out_w.write('%d %g\n' % (k*n + i, w * w_sign))
    for (i,u) in enumerate(weights_u):
      if u != 0:
        out_u.write('%d %s\n' % (k*n + i + nw, u))

  out_w.close()
  out_u.close()
  
  print('Wrote ' + fname_out_w)
  print('Wrote ' + fname_out_u)

if __name__ == '__main__':
  main()

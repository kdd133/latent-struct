#!/usr/bin/env python

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Copyright (c) 2013 Kenneth Dwyer

import optparse
import os
import random
import re
import sys
from math import ceil, exp, log

class Hyperparameter:
  def __init__(self, name, type, vals):
    self.name = name
    if type not in ('f', 'i', 'b', 's', 'l'):
      raise ValueError, 'The type qualifier must be f(float), i(int), b(bool), \
s(set), or l(logarithmic).'
    self.type = type
    self.vals = vals
    
  def sample(self):
    if self.type == 'f':   # float
      assert(len(self.vals) == 2)
      lower, upper = float(self.vals[0]), float(self.vals[1])
      # Note: The uniform function swaps the arguments if lower >= upper.
      return random.uniform(lower, upper)
    elif self.type == 'i': # integer
      assert(len(self.vals) == 2)
      lower, upper = int(self.vals[0]), int(self.vals[1])
      if lower >= upper:
        raise ValueError, 'Hyperparameter %s has lower >= upper.' % self.name
      return random.randint(lower, upper)
    elif self.type == 'b': # boolean
      # Simply ignore the vals field in this case.
      return random.choice((True, False))
    elif self.type == 's': # set
      assert(len(self.vals) >= 2)
      return random.choice(self.vals)
    elif self.type == 'l': # logarithmic
      assert(len(self.vals) == 2)
      lower, upper = log(float(self.vals[0])), log(float(self.vals[1]))
      # Note: The uniform function swaps the arguments if lower >= upper.
      return exp(random.uniform(lower, upper))
    else:
      raise ValueError, 'An unrecognized parameter type was encountered.'

usage = 'Usage: %prog [options] PARAM1:[TYPE1:]VAL1a[,VAL1b,...] \
[PARAM2:[TYPE2:]VAL2a[,VAL2b,...]]\n\n\
  This program generates a shell script that runs latent_struct multiple\n\
  times using different hyperparameter settings. Hyperparameters are specified\n\
  here in two ways: as fixed values or as random values. In the case of a fixed\n\
  value, the form used is PARAM:VALUE. In the case of a random value, the form\n\
  used is PARAM:TYPE:VAL1[,VAL2,...], where type is f(float), i(integer),\n\
  b(boolean), or s(set). More specifically:\n\
    f(float): there must be exactly two values, e.g., beta:f:0.1,0.5 will assign\n\
      a value to beta that is drawn uniformly from [0.1,0.5]\n\
    i(integer): like f(float), except that the values must be integers\n\
    b(boolean): there should not be any values (they will be ignored), e.g.,\n\
      align-unigrams-only:b:\n\
    s(set): there must be two or more values, e.g.,\n\
      weights-init:s:heuristic,noise,zero uniformly chooses one of three values\n\
    l(logarithmic): like f(float), except that the range is transformed into\n\
      log-space, a sample is drawn uniformly from the resulting range, and\n\
      the value is exponentiated'

parser = optparse.OptionParser(usage=usage)
parser.add_option('-e', '--executable', type='string', default='./latent_struct',
                  help='command to execute (default: ./latent_struct)')
parser.add_option('--script-seed', type='int', default=0,
                  help='seed for the script\'s random number generator (default: 0)')
parser.add_option('--train', type='string', default=None,
                  help='name of file containing training data (default: None)')
parser.add_option('-j', '--num-jobs', type='int', default=1,
                  help='number of jobs to generate (default: 1)')
parser.add_option('-n', '--num-trials', type='int', default=10,
                  help='number of hyperparameter settings to try (default: 10)')
parser.add_option('-p', '--prefix', type='string', default=None,
                  help='an identifier for the experiment (default: None)')
parser.add_option('--dir', type='string', default='.',
                  help='base directory for results (default: .)')
parser.add_option('--shuffle', action='store_true', default=False,
                  help='shuffle the order of the commands')
opts, args = parser.parse_args()

if len(args) < 1:
  print 'Error: At least one option-value pair (i.e., PARAM:VAL1) is required.\n'
  parser.print_help()
  sys.exit(-1)
  
if not (opts.train and opts.prefix and opts.script_seed):
  print 'One or more of the required options has value None.'
  print 'The required options are --train, --prefix, and --script-seed.'
  parser.print_help()
  sys.exit(-1)
  
random.seed(int(opts.script_seed))

results_dir = opts.dir.rstrip('/')

got_model = False
got_obj = False
got_opt = False

comma = re.compile('.*,')

fixed = []
to_sample = []

for arg in args:
  fields = arg.split(':')
  
  if len(fields) < 2 or len(fields) > 3:
    print 'Each argument should have either 2 or 3 colon-separated fields\n  ' \
      + arg
    sys.exit(-1)

  if len(fields) == 2:
    # This is a fixed parameter that will not be randomized.
    param, value = fields
    if comma.match(value):
      print 'Parameters that are fixed (i.e., do not have a type qualifier) \
may not take multiple values:\n  ' + arg
      sys.exit(-1)
    if param == 'model': got_model = True
    if param == 'obj': got_obj = True
    fixed.append((param, value))
    
  elif len(fields) == 3:
    # This is a parameter that we want to randomize.
    param, type, values = fields
    to_sample.append(Hyperparameter(param, type, values.split(',')))

if not (got_model and got_obj):
  print 'Error: Option-value pairs for model and obj are required.'
  sys.exit(1) 

commands = []
for n in xrange(opts.num_trials):
  dir_n = '%s/%s/%03d' % (results_dir, opts.prefix, n)
  options = ['--dir=%s' % dir_n, '--train=%s' % opts.train]
  options.extend('--%s=%s' % (p,v) for p,v in fixed)
  for param in to_sample:
    arg = param.name
    val = param.sample()
    # the short forms of beta (-b) and tolerance (-t) get special treatment
    if arg == 'b' or arg == 't':
      options.append('-%s%s' % (arg, val))
    # boolean switches are included if true, omitted if false
    elif param.type == 'b':
      if val == True:
        options.append('--%s' % arg)
    # all others are included in long format
    else:
      options.append('--%s=%s' % (arg, val))

  cmd = 'if [ ! -e %s ]; then mkdir -p %s && %s %s >& %s/output && gzip %s/*; fi\n' \
    % (dir_n, dir_n, opts.executable, ' '.join(sorted(options)), dir_n, dir_n)
  commands.append(cmd)

#FIXME: It is possible for the last job to be allocated zero commands.
cmds_per_job = int(ceil(len(commands)/float(opts.num_jobs)))
if opts.shuffle:
  random.shuffle(commands)
it = commands.__iter__()
for n in range(opts.num_jobs):
  fname = '%s_%03d.sh' % (opts.prefix, n)
  if os.path.exists(fname):
    print 'Error: Script %s already exists' % fname
    sys.exit(1)
  f = open(fname, 'w')
  f.write('#!/bin/bash\n')
  f.write('if [ "$PBS_O_WORKDIR" != "" ]; then\n  cd $PBS_O_WORKDIR\nfi\n')
  f.write('\n# %s\n\n' % ' '.join(sys.argv))
  for _ in range(cmds_per_job):
    try:
      f.write(it.next() + '\n')
    except:
      break  # we've used up all the commands
  f.write('\nexit 0\n')
  print 'Wrote ' + fname
  try:
    os.fchmod(f.fileno(), 0500)
  except Exception:
    print '  Please run "chmod +x %s" to make the script executable' % fname
  f.close()


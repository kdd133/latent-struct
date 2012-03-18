#!/usr/bin/env python

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Copyright (c) 2012 Kenneth Dwyer

import math
import optparse
import os
import random
import re
import sys

# Source: http://goo.gl/OG2KO
permute = lambda ss,row=[],level=0: len(ss)>1 \
  and reduce(lambda x,y:x+y,[permute(ss[1:],row+[i],level+1) for i in ss[0]]) \
  or [row+[i] for i in ss[0]]
  
sep = ','

usage = 'Usage: %prog [options] PARAM1:VAL1[,VAL2,...] [PARAM2:VAL1[,VAL2,...]]\n\n\
  This script is used to run latent_struct several times with different parameter\n\
  settings, i.e., to perform a grid search over some range of values for a subset\n\
  of the available hyper-parameters. For example, to tune the regularization\n\
  constant beta, we might include an argument of the form beta:1e-1,1e-2,1e-3.\n\
  Note that boolean options, which are passed to latent_struct as flags can be\n\
  enabled or disabled via %prog using <flagname>:true and <flagname>:false\n\
  (e.g., --em is toggled by including the argument em:true,false). Parameters\n\
  that are left unspecified will take their default values in latent_struct.'
parser = optparse.OptionParser(usage=usage)
parser.add_option('-e', '--executable', type='string', default='./latent_struct',
                  help='command to execute (default: ./latent_struct)')
parser.add_option('--train', type='string', default=None,
                  help='name of file containing training data (default: None)')
parser.add_option('-j', '--num-jobs', type='int', default=1,
                  help='number of jobs to generate (default: 1)')
parser.add_option('-p', '--prefix', type='string', default=None,
                  help='an identifier for the experiment (default: None)')
parser.add_option('--dir', type='string', default='.',
                  help='base directory for results (default: .)')
parser.add_option("-s", '--shuffle', action='store_true', default=False,
                  help='shuffle the order of the commands')
opts, args = parser.parse_args()

if len(args) < 1:
  print 'Error: At least one option-value pair (i.e., PARAM:VAL1) is required.\n'
  parser.print_help()
  sys.exit(-1)
  
if not (opts.train and opts.prefix):
  print 'One or more of the required options has value None.'
  parser.print_help()
  sys.exit(-1) 

results_dir = opts.dir.rstrip('/')

got_model = False
got_obj = False
got_opt = False

settings = []
for gridparam in args:
  param = gridparam.split(':')[0]
  if param == 'model': got_model = True
  if param == 'obj': got_obj = True
  settings.append(['%s:%s' % (param,v) for v in \
                   gridparam.split(':')[1].split(sep)])

if not (got_model and got_obj):
  print 'Error: Option-value pairs for model and obj are required.'
  sys.exit(1) 

combinations = permute(settings)
commands = []
space = re.compile(' ')
for n in xrange(len(combinations)):
  comb = combinations[n]
  dir_n = '%s/%s/%03d' % (results_dir, opts.prefix, n)
  required = '--dir=%s --train=%s' % (dir_n, opts.train)
  options = []
  for param in comb:
    arg, val = tuple(param.split(':'))
    # the short forms of beta (-b) and tolerance (-t) get special treatment
    if arg == 'b' or arg == 't':
      options.append('-%s%s' % (arg,val))
    # boolean switches are included if true, omitted if false
    elif val == 'true':
      options.append('--%s' % arg)
    elif val == 'false':
      continue
    # all other options are included in long format
    else:
      options.append('--%s=%s' % (arg, val))

  cmd = 'if [ ! -e %s ]; then mkdir -p %s && %s %s %s >& %s/output && gzip %s/*; fi\n' \
    % (dir_n, dir_n, opts.executable, required, ' '.join(options), dir_n, dir_n)
  commands.append(cmd)

#FIXME: It is possible for the last job to be allocated zero commands.
cmds_per_job = int(math.ceil(len(commands)/float(opts.num_jobs)))
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


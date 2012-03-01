#!/usr/bin/env python

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Copyright (c) 2012 Kenneth Dwyer

import gzip
import optparse
import re
import sys

def extract_fields(f, fields, values, prepend='', index=1):
  for line in f.readlines():
    for (field_name, regex) in fields:
      if regex.match(line):
        values[prepend+field_name] = float(line.rstrip().split()[index])
        
def read_options(s, results):
  results['flags'] = ''
  for o in s.split():
    try:
      option, value = o.split('=')
      option = option[2:] # remove the -- that's prepended to the option name    
      results[option] = value
    except ValueError:
      results['flags'] += o[2:] + ' '    
  
def results_html(fout, results):
  # map cryptic names to "pretty" names; columns will be printed in this order  
  stat_names = {'Train-Accuracy' : 'Train acc',
                'Train-Precision' : 'Train prec',
                'Train-Recall' : 'Train rec',
                'Train-Fscore' : 'Train F1',
                'Eval-Accuracy' : 'Eval acc',
                'Eval-Precision' : 'Eval prec',
                'Eval-Recall' : 'Eval rec',
                'Eval-Fscore' : 'Eval F1'}
  
  option_names_to_skip = ('dir',)
  
  fout.write('<table id="table" class="tablesorter">\n')
  fout.write('<thead align="center">\n  <tr>\n')
  fout.write('    <th>ID</th>\n')
  for statname in sorted(stat_names):
    fout.write('    <th>%s</th>\n' % stat_names[statname])
  # Note: this assumes that the same options appear in all jobs
  for name in sorted(results[min(results)]):
    if name not in stat_names and name not in option_names_to_skip:
      fout.write('    <th>%s</th>\n' % name)
  fout.write('  </tr>\n</thead>\n\n')
  
  fout.write('<tbody align="right">\n')
  for subfolder in results:
    fout.write('  <tr>\n')
    fout.write('    <td>%s</td>\n' % subfolder.split('/')[-1])
    for statname in sorted(stat_names):
      fout.write('    <td>%.3f</td>\n' % results[subfolder][statname])
    for name in sorted(results[subfolder]):
      if name not in stat_names and name not in option_names_to_skip:
        fout.write('    <td>%s</td>\n' % results[subfolder][name])
    fout.write('  </tr>\n\n')
  fout.write('</tbody>\n\n')
  
###################

if __name__ == '__main__':
  usage = 'Usage: %prog [options] EXPERIMENT_SUBDIR_0 [EXPERIMENT_SUBDIR_1] ...\n\n\
    This script extracts statistics from the output files produced by one or more\n\
    runs of latent_struct, and aggregates them into a single html file that contains\n\
    a large table, which can be sorted according to a chosen column.\n\n\
    A typical call would include all the sub-directories created by executing a script\n\
    produced by grid.py, e.g., %prog <results_dir>/*, where <results_dir> is the path\n\
    that was provided as the --dir option to grid.py.\n\n\
    Note: The table in the resulting html file will be sortable (and formatted nicely),\n\
    provided that you download the following files and place them in the same directory\n\
    as the html file (the latter zip file must also be extracted):\n\
      http://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js\n\
      http://autobahn.tablesorter.com/jquery.tablesorter.min.js\n\
      http://tablesorter.com/themes/blue/blue.zip'

  parser = optparse.OptionParser(usage=usage)
  opts, args = parser.parse_args()

  if len(args) == 0:
    parser.print_help()
    sys.exit()

  field_names = ('Train-Accuracy', 'Train-Precision', 'Train-Recall',
                 'Train-Fscore', 'Eval-Accuracy', 'Eval-Precision',
                 'Eval-Recall', 'Eval-Fscore', '11-pt')
  field_res = []
  for field in field_names:
    field_res.append(re.compile('.*%s' % field))
  fields = zip(field_names, field_res)

  results = {}
  for folder_name in args:
    folder_name = folder_name.rstrip('/')
    results[folder_name] = {}
    
    output_fname = '%s/%s' % (folder_name, 'output.gz')
    fin = gzip.open(output_fname)
    extract_fields(fin, fields, results[folder_name])
    fin.close()
        
    options_fname = '%s/%s' % (folder_name, 'options.txt.gz')
    fin = gzip.open(options_fname)
    read_options(fin.readline().rstrip(), results[folder_name])
    fin.close()
    
    # Get the 11-pt avg precision for train and eval if the files are present
    train_avgprec_fname = '%s/%s' % (folder_name, 'train_11pt_avg_prec.txt')
    try:
      fin = open(train_avgprec_fname)
      extract_fields(fin, fields, results[folder_name], 'Train ', 2)
      fin.close()
    except IOError:
      pass    
    eval_avgprec_fname = '%s/%s' % (folder_name, 'eval_11pt_avg_prec.txt')
    try:
      fin = open(eval_avgprec_fname)
      extract_fields(fin, fields, results[folder_name], 'Eval ', 2)
      fin.close()
    except IOError:
      pass
  
  experiment_name = args[0].split('/')[0]
  html_name = '%s.html' % experiment_name
  fout = open(html_name, 'w')
  fout.write('<html>\n')
  fout.write('<head>\n')
  fout.write('<title>%s results</title>\n' % experiment_name)
  fout.write('<link rel="stylesheet" href="themes/blue/style.css" type="text/css" id="" media="print, projection, screen" />\n')
  fout.write('<script type="text/javascript" src="jquery.min.js"></script>\n')
  fout.write('<script type="text/javascript" src="jquery.tablesorter.min.js"></script>\n')
  fout.write('<script type="text/javascript" id="js">$(document).ready(function() {\n')
  fout.write('    $("table").tablesorter({sortList: [[0,0]]});\n')
  fout.write('  }\n')
  fout.write(');\n')
  fout.write('</script>\n')
  fout.write('</head>\n')
  fout.write('<body style="font-family:sans-serif;">\n')
  
  fout.write('<h1>Results: %s</h1>\n' % experiment_name)
  fout.write('<p>\nYou can click on a column header in the table to sort by that field. ' \
             'Moreover, you can hold the shift key (at least in firefox) and then click ' \
             'on additional columns in order to sort according to several fields. ' \
             '\n</p>\n')
  results_html(fout, results) # writes the tabular data to fout
  
  fout.write('</body>\n')
  fout.write('</html>\n')
  fout.close()

  print 'Wrote ' + html_name


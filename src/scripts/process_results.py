#!/usr/bin/env python

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Copyright (c) 2012 Kenneth Dwyer

import glob
import gzip
import optparse
import re
import sys

# labels of fields to (attempt to) extract from the output files
field_names = ('^([0-9]+)-(Objective-Value)',
               '^([0-9]+)-(Train-Accuracy)',
               '^([0-9]+)-(Train-Precision)',
               '^([0-9]+)-(Train-Recall)',
               '^([0-9]+)-(Train-Fscore)',
               '^([0-9]+)-(Train-11ptAvgPrec)',
               '^([0-9]+)-(Eval-Accuracy)',
               '^([0-9]+)-(Eval-Precision)',
               '^([0-9]+)-(Eval-Recall)',
               '^([0-9]+)-(Eval-Fscore)',
               '^([0-9]+)-(Eval-11ptAvgPrec)',
               '^([0-9]+)-(beta)',
               '^([0-9]+)-(tolerance)',
               '^([0-9]+)-(status)',
               '^ *(11-pt) avg')

# map cryptic names to "pretty" names; columns will be printed in this order  
stat_names = {'Objective-Value': 'Obj',
              'Train-Accuracy' : 'Train acc',
              'Train-Precision' : 'Train prec',
              'Train-Recall' : 'Train rec',
              'Train-Fscore' : 'Train F1',
              'Train-11ptAvgPrec' : 'Train 11pt',
              'Eval-Accuracy' : 'Eval acc',
              'Eval-Precision' : 'Eval prec',
              'Eval-Recall' : 'Eval rec',
              'Eval-Fscore' : 'Eval F1',
              'Eval-11ptAvgPrec' : 'Eval 11pt'}

# these options will be omitted from any output
option_names_to_skip = ('dir',)


def extract_fields(f, fields, results, id, prepend='', index=1):
  for line in f.readlines():
    for (field_name, regex) in fields:
      m = regex.search(line)
      if m:
        if len(m.groups()) == 2:
          key = '%s-%s' % (id, m.group(1))
          if key not in results:
            results[key] = {}
          stat = prepend + m.group(2)
          results[key][stat] = float(line.rstrip().split()[index])
        elif len(m.groups()) == 1:
          if id not in results:
            results[id] = {}
          stat = prepend + m.group(1)
          results[id][stat] = float(line.rstrip().split()[index])
        else:
          raise Exception('An unexpected number of groups was encountered.')

def read_options(s, results, id):
  opts = {}
  opts['flags'] = ''
  grid_options = re.compile('^\-([bt]|\-beta|\-tolerance)')
  for o in s.split():
    if grid_options.match(o):
      continue
    try:
      option, value = o.split('=')
      option = option[2:] # remove the -- that's prepended to the option name    
      opts[option] = value
    except ValueError:
      opts['flags'] += o[2:] + '+'

  if len(opts['flags']) > 0:
    opts['flags'] = opts['flags'][:-1] # drop trailing '+' sign

  # the options are the same for a given id/subfolder (e.g., results/001), so
  # we copy them to the sub id's (e.g., results/001-0, results/001-1) that
  # correspond to this "primary" id
  id = re.sub(r'\+', '\\+', id) # '+' in the filename/id needs to be escaped
  id_re = re.compile(id + '\-[0-9]+')
  for sub_id in results.keys():
    if id_re.match(sub_id):
      for option, value in opts.items():
        results[sub_id][option] = value    
  
def results_txt(fout, results):
  fout.write('ID\t')
  fout.write('\t'.join([s for s in sorted(stat_names)]))
  # Note: this assumes that the same options appear in all jobs
  for name in sorted(results[min(results)]):
    if name not in stat_names and name not in option_names_to_skip:
      fout.write('\t%s' % name)
  fout.write('\t\n')
  
  for subfolder in results:
    fout.write('%s\t' % subfolder.split('/')[-1])
    for statname in sorted(stat_names):
      try:
        fout.write('%.3f\t' % results[subfolder][statname])
      except KeyError:
        fout.write('N/A\t')
    for name in sorted(results[subfolder]):
      if name not in stat_names and name not in option_names_to_skip:
        try:
          fout.write('%s\t' % results[subfolder][name])
        except KeyError:
          fout.write('N/A\t')
    fout.write('\n')

def results_html(fout, results, experiment_name = "latent_struct experiment",):
  fout.write('<html>\n')
  fout.write('<head>\n')
  fout.write('<title>%s results</title>\n' % experiment_name)
  fout.write('<link rel="stylesheet" href="themes/blue/style.css" \
type="text/css" id="" media="print, projection, screen" />\n')
  fout.write('<script type="text/javascript" src="jquery.min.js"></script>\n')
  fout.write('<script type="text/javascript" src="jquery.tablesorter.min.js">\
</script>\n')
  fout.write('<script type="text/javascript" id="js">$(document).ready(\
function() {\n')
  fout.write('    $("table").tablesorter({sortList: [[0,0]]});\n')
  fout.write('  }\n')
  fout.write(');\n')
  fout.write('</script>\n')
  fout.write('</head>\n')
  
  fout.write('<body style="font-family:sans-serif;">\n')    
  fout.write('<h1>Results: %s</h1>\n' % experiment_name)
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
      try:
        fout.write('    <td>%.3f</td>\n' % results[subfolder][statname])
      except KeyError:
        fout.write('    <td>N/A</td>\n')
    for name in sorted(results[subfolder]):
      if name not in stat_names and name not in option_names_to_skip:
        try:
          fout.write('    <td>%s</td>\n' % results[subfolder][name])
        except KeyError:
          fout.write('    <td>N/A</td>\n')
    fout.write('  </tr>\n\n')
  fout.write('</tbody>\n\n')
  
  fout.write('</body>\n')
  fout.write('</html>\n')
  
###################

if __name__ == '__main__':
  usage = '%prog [options] EXPERIMENT_SUBDIR_0 [EXPERIMENT_SUBDIR_1] ...\n\n\
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

  field_res = []
  for field in field_names:
    field_res.append(re.compile(field))
  fields = zip(field_names, field_res)

  results = {}
  for folder_name in args:
    folder_name = folder_name.rstrip('/')

    output_fname = '%s/%s' % (folder_name, 'output.gz')
    try:
      fin = gzip.open(output_fname)
    except IOError:
      print 'Skipping incomplete result in %s' % folder_name
      continue;
    fin = gzip.open(output_fname)
    extract_fields(fin, fields, results, folder_name)
    fin.close()
        
    options_fname = '%s/%s' % (folder_name, 'options.txt.gz')
    fin = gzip.open(options_fname)
    read_options(fin.readline().rstrip(), results, folder_name)
    fin.close()
    
    # Get the 11-pt avg precision for train and eval if the files are present
    try :
      wild = glob.glob('%s/*-%s' % (folder_name, '*_11pt_avg_prec.txt.gz'))
      for fname in wild:
        fin = gzip.open(fname)
        id = '%s-%s' % (folder_name, (fname.split('/')[-1]).split('-')[0])
        if re.search('train_11pt_avg_prec', fname):
          label = 'Train '
        else:
          label = 'Eval '
        extract_fields(fin, fields, results, id, label, 2)
        fin.close()
    except IOError:
      pass

  experiment_name = '/'.join(args[0].split('/')[:-1])

  # output a table of results to a plain text file
  txt_name = '%s.txt' % experiment_name
  fout = open(txt_name, 'w')
  results_txt(fout, results)
  fout.close()
  print 'Wrote ' + txt_name

  # output a table of results to a html file
  html_name = '%s.html' % experiment_name
  fout = open(html_name, 'w')  
  results_html(fout, results, experiment_name)
  fout.close()
  print 'Wrote ' + html_name

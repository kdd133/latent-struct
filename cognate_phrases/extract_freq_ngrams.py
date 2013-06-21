import codecs
import optparse
import sys
from ngram_cdf import *

"""
Takes as input one or more ngram files (output by SRILM ngram-count) and
writes to the specified output file only those ngrams whose frequency is above
the given percentile (--cut-fraction). If the --top-k option is provided, the
program instead outputs the k most frequent n-grams from each file, in which
case --cut-fraction is ignored.
"""

def main():
    usage = 'Usage: %prog [options] <file1> [file2] ...'
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('-c', '--cut-fraction', type='float', default=None,
                      help='discard the n-grams that are below this percentile in terms of frequency')
    parser.add_option('-k', '--top-k', type='int', default=None,
                      help='write the k most frequent n-grams from each file')
    parser.add_option('-o', '--output', type='string', default=None,
                      help='name of the output file (default: None)')
    opts, args = parser.parse_args()
    
    if len(args) < 1:
        print('Error: At least one input file is required.')
        parser.print_help()
        return
    
    if not opts.output:
        print('Error: You must specify an output file.')
        parser.print_help()
        return
    
    if not (opts.cut_fraction or opts.top_k):
        print('Error: Either --cut-fraction or --top-k must be specified.')
        parser.print_help()
        return
    
    if opts.top_k:
        print('using --top-k=%d' % opts.top_k)
    else:
        print('using --cut-fraction=%g' % opts.cut_fraction)
    
    fout = codecs.open(opts.output, 'w', encoding='latin1')        
    for fname in args:
        f = codecs.open(fname, 'r', encoding='latin1')
        if opts.top_k:
            writeMostFrequentNgrams(f, fout, opts.top_k)
        else:
            counts, cumul = getCumulativeCounts(f)
            min_count = getCountForFraction(counts, cumul, opts.cut_fraction)
            print('min_count for %s is %d' % (fname, min_count))
            writeNgramsWithMinCount(f, fout, min_count)
        f.close()
    fout.close()

def writeNgramsWithMinCount(fin, fout, min_count):
    fin.seek(0) # rewind
    for raw_line in fin.readlines():
        line = raw_line.rstrip()
        if len(line) == 0:
            continue
        count = int(line.split()[-1])
        if count >= min_count:
            fout.write(raw_line)
            
def writeMostFrequentNgrams(fin, fout, k):
    fin.seek(0) # rewind
    freq = []
    for raw_line in fin.readlines():
        line = raw_line.rstrip()
        if len(line) == 0:
            continue
        count = int(line.split()[-1])
        freq.append((count, raw_line))
        
    freq.sort(reverse=True)
    for count, raw_line in freq[:k]:
        fout.write(raw_line)

if __name__ == "__main__":
    main()

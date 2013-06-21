import codecs
import sys
from matplotlib import pyplot

"""
Takes as input a file written by the SRILM ngram-count program. Produces a plot
of the cumulative distribution function of the n-gram frequencies. If the second
argument "cut fraction" is provided, the program does not produce a plot, but
instead prints the count/frequency below which the given fraction of n-grams
occur.
"""

def getCountForFraction(counts, cumulative, cut):
    assert cut > 0 and cut < 1
    total = float(cumulative[-1])
    for c,accum in zip(counts, cumulative):
        if accum/total > cut:
            return c

def getCumulativeCounts(f):
    f.seek(0) # rewind
    freq_counts = {}
    for line in f.readlines():
        line = line.rstrip()
        if len(line) == 0:
            continue
        count = int(line.split()[-1])
        if count not in freq_counts:
            freq_counts[count] = 0
        freq_counts[count] += 1
    
    counts = sorted(freq_counts.keys())
    n = len(counts)
    cumulative = [0]*n
    accum = 0
    for i,c in zip(range(n), counts):
        cumulative[i] = accum + freq_counts[c]
        accum += freq_counts[c]

    return (counts, cumulative)


def main():
    if len(sys.argv) < 2:
        print('Usage: %s <ngram-count output file> [cut fraction]' % sys.argv[0])
        return
    
    f = codecs.open(sys.argv[1], 'r', encoding='latin1')
    counts, cumulative = getCumulativeCounts(f)
    f.close()

    if len(sys.argv) < 3:
        pyplot.plot(counts, cumulative)
        pyplot.xlabel('Cumulative frequency')
        pyplot.ylabel('N-gram count')
        pyplot.xscale('log')
        pyplot.show()
    else:
        cut = float(sys.argv[2])
        min_count = getCountForFraction(counts, cumulative, cut)
        print('%g percent of n-grams have count <= %d' % (100*cut, min_count))

if __name__ == "__main__":
    main()

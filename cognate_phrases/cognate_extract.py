import codecs
import sys

"""
Extracts the unique source or target words from the given cognate pairs file,
and prints the words to the output file with characters separated by spaces.
"""

def main():
    if len(sys.argv) < 4:
        print('Usage: %s <cognate input file> <output file> <source|target>' % sys.argv[0])
        return
    
    to_extract = sys.argv[3]
    if to_extract == 'source':
        extract_field = 1
    elif to_extract == 'target':
        extract_field = 2
    else:
        print('Error: The second argument must be "source" or "target".')
        return
    
    f = codecs.open(sys.argv[1], 'r', encoding='latin1')
    words = set()
    for line in f.readlines():
        line = line.rstrip()
        if len(line) == 0:
            continue
        fields = line.split()
        words.add(fields[extract_field])
    f.close()
    
    f = codecs.open(sys.argv[2], 'w', encoding='latin1')
    for word in sorted(words):
        f.write(' '.join(list(word)) + '\n')
    f.close()
    
if __name__ == "__main__":
    main()

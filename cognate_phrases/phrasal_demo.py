import sys

def print_window(s, t, sw, tw):
    #print(sw, tw) # for debugging
    print(''.join([s[i] for i in sw]), ''.join([t[j] for j in tw]))

def contains_match(win, match):
    for i in win:
        if match[i]:
            return True
    return False

def matches_consistent(s, t, sw, tw, match):
    #print_window(s, t, sw, tw)
    for i in sw:
        if match[i] and i not in tw:
            return False
    for j in tw:
        if match[j] and j not in sw:
            return False    
    return True

def print_features(s, t, sk, tk):
    assert len(s) == len(t)
    match = [s_char == t_char for (s_char,t_char) in zip(s,t)]
    eps = '-'
    length = len(s)
    
    for pos in range(length):
        if s[pos] == eps:
            continue
        s_win = []
        i = pos
        j = -1
        # get the initial source window
        while len(s_win) < sk and i < length:
            if s[i] != eps:
                s_win.append(i)
                if match[i] and j < 0:
                    j = i
            i += 1
        
        # we don't have a window of the desired width at this position
        if len(s_win) < sk:
            continue
        
        # there are no matches in the source window
        if j < 0:
            continue
        
        # grow the initial target window left from the first match in s_win
        t_win = []
        while len(t_win) < tk and j >= 0:
            if t[j] != eps:
                t_win.insert(0, j)
            j -= 1
        
        # grow the target window up to width tk
        j = t_win[-1]+1
        while len(t_win) < tk and j < length:
            if t[j] != eps:
                t_win.append(j)
            j += 1
        
        if matches_consistent(s, t, s_win, t_win, match):        
            print_window(s, t, s_win, t_win)
        
        j = t_win[-1]+1
        while j < length:
            if match[j] and j not in s_win:
                break
            if t[j] != eps:
                t_win.append(j)
                t_win.pop(0)
                if contains_match(t_win, match):
                    if matches_consistent(s, t, s_win, t_win, match):
                        print_window(s, t, s_win, t_win)
            j += 1

def main():
    max_phrase_len = 3

    if len(sys.argv) < 3:
        print('Example usage: %s ^-pon-osi-t$ ^c-onjo-int$' % sys.argv[0])
        return
    
    src = sys.argv[1]
    tgt = sys.argv[2]
    
    #print(' '.join(list(src)))
    #print(' '.join(list(tgt)))
    #print()
    
    assert len(src) == len(tgt)
    length = len(src)
    
    for sk in range(1, max_phrase_len+1):
        for tk in range(1, max_phrase_len+1):
            #print('sk=%d tk=%d' % (sk,tk)) # for debugging
            print_features(src, tgt, sk, tk)

main()

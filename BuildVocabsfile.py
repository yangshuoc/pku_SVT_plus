import os
import re
import numpy as np
import random
from collections import Counter

CLEAN_DIR = 'Clean_Data'
VOCAB_FILE = 'vocab_full.txt'

def open_file(filename, mode='r'):
    # return open(filename, mode=mode, encoding='utf-8', errors='ignore')
    return open(filename, mode=mode, encoding='ascii', errors='ignore')

def build_vocab():
    all_words = []
    for fname in sorted(os.listdir(CLEAN_DIR)):
        #是否去掉'?
        words = open_file(os.path.join(CLEAN_DIR, fname)).read().strip().replace('\n', ' ').split()
        for word in words:
            if word.find("'") == 0:
                continue
            all_words.append(word)
    count_pairs = Counter(all_words)
    # count_pairs = count_pairs.most_common(VOCAB_SIZE)
    count_pairs = count_pairs.most_common(len(count_pairs))
    #总词数有223917
    print("vocabs number:")
    print(len(count_pairs))
    words, _ = list(zip(*count_pairs))
    open_file(VOCAB_FILE, 'w').write('\n'.join(words) + '\n')

   

#python BuildVocabsfile.py
if __name__ == '__main__':
    build_vocab()


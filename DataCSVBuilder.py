import numpy as np
import os
import csv

CLEAN_DIR = 'Clean_Data'
VOCAB_FILE = 'vocab_full.txt'
VOCAB_SIZE = 90
WORD_SP = ' '
VACAB_MAP = {}
CSV_FILE = 'vocab_vector.csv'

def open_file(filename, mode='r'):
    # return open(filename, mode=mode, encoding='utf-8', errors='ignore')
    return open(filename, mode=mode, encoding='ascii', errors='ignore')

def getVocabs():
    VocabsFull = open_file(VOCAB_FILE).readlines()
    Vocabs = VocabsFull[:VOCAB_SIZE]
    result = []
    for v in Vocabs:
        v = v.replace('\n','')
        result.append(v)
    return result

def initVacabMap(vacabs):    
    for i in range(len(vacabs)):
        VACAB_MAP[vacabs[i]] = i

def findIndex(var):
    try:
        i = VACAB_MAP[var]
        return i
    except KeyError:
        return -1
def buildMatrix(vocabs):
    matrix = []
    for fname in sorted(os.listdir(CLEAN_DIR)):
        print("!!!!!!!!!!!!!!!!!"+"processing file "+ fname + "!!!!!!!!!!!!!!!!!!!!!!!")
        lines = open_file(os.path.join(CLEAN_DIR, fname)).readlines()
        matrixLine = np.ones(len(vocabs))
        for line in lines:
            words = line.split(WORD_SP)
            for word in words:
                index = findIndex(word)
                if index != -1:
                    matrixLine[index] += 1
        sum = np.sum(matrixLine)
        matrixLine =  (matrixLine/sum)*1
        matrix.append(matrixLine.tolist())
    return matrix
def saveCSV(vocabs,matrix):
    writer = csv.writer(open(CSV_FILE, mode='w',encoding='utf-8',newline=''))
    writer.writerow(vocabs)
    for m in matrix:
        writer.writerow(m)

#python DataCSVBuilder.py
if __name__ == '__main__':
    vocabs = getVocabs()
    initVacabMap(vocabs)
    matrix = buildMatrix(vocabs)
    print("matrix rank:")
    print(np.linalg.matrix_rank(matrix))
    saveCSV(vocabs,matrix)
    print("vocab vector saved to vocab_vector.csv")



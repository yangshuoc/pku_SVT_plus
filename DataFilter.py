import csv
import random

# DATA_FILE = 'data.csv'
# MISS_TAG = 0
# MISSING_ROW_NUM = 900
# MISSING_COL_NUM = 6

DATA_FILE = 'vocab_vector.csv'
MISS_TAG = 0
MISSING_ROW_NUM = 200
# MISSING_COL_NUM = 70
MISSING_COL_NUM = 50

# DATA_FILE = 'svt_matrix.csv'
# MISS_TAG = 0
# MISSING_ROW_NUM = 70
# MISSING_COL_NUM = 250

def loadCSVData(file):
    csv_reader = csv.reader(open(file, encoding='utf-8'))
    strMatrix = []
    for r in csv_reader:
        strMatrix.append(r)
    numMatrix = []
    for i in range(len(strMatrix)):
        if i == 0:
            continue
        row = []
        for x in strMatrix[i]:
            row.append(float(x))
        numMatrix.append(row)
    return numMatrix

def buildTestMatrix(matrix):
    n = len(matrix)
    m = len(matrix[0])
    selectedN = range(n)
    selectedM = range(m)
    selectedN = random.sample(selectedN,n-MISSING_ROW_NUM)
    selectedM = random.sample(selectedM,m-MISSING_COL_NUM)
    
    selectedN.sort()
    selectedM.sort()
    # print(selectedM)
    # print(selectedN)

    for i in range(n):
        for j in range(m):
            if i in selectedN or j in selectedM:
                continue
            matrix[i][j] = MISS_TAG
    return selectedN,selectedM,matrix

def getCsrMatrix(M,row,col):
    n = len(M)
    m = len(M[0])
    vector = []
    for i in range(n):
        for j in range(m):            
            if M[i][j] != MISS_TAG:
                row.append(i)
                col.append(j)
                vector.append(M[i][j])
    # print("finished")
    return vector,n,m

if __name__ == '__main__':
    matrix = loadCSVData(DATA_FILE)
    selectedN,selectedM,matrix = buildTestMatrix(matrix)
    
    row = []
    col = []
    vector = getCsrMatrix(matrix,row,col) 
    # print(vector)

    for row in matrix:
        print(row)

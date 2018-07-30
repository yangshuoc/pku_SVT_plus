import numpy as np
import DataFilter as df
import scipy.sparse as ss
from sparsesvd import sparsesvd

SIMILAR_RANK = 10

def get_full_M():
    return df.loadCSVData(df.DATA_FILE)

def get_M():
    matrix = get_full_M()
    selectedN,selectedM,matrix = df.buildTestMatrix(matrix)
    return selectedN,selectedM,matrix

def get_A(M,selected_col):
    M = np.transpose(M)
    C = []
    for j in range(len(M)):
        if j in selected_col:
            C.append(M[j])
    A = np.mat(np.transpose(C))    
    return A

def get_B(M,selected_row):
    R = []
    for i in range(len(M)):
        if i in selected_row:
            R.append(M[i])
    B = np.mat(np.transpose(R))
    return B
def get_feature_vector(Y):
    ut, s, vt = sparsesvd(ss.csc_matrix(Y),SIMILAR_RANK)
    return ut
   
def get_AAT(M,selected_col):
    M = np.transpose(M)
    C = []
    for j in range(len(M)):
        if j in selected_col:
            C.append(M[j])
    A = np.mat(np.transpose(C))
    print(np.shape(A))
    At = C
    AAT = A*At
    return AAT

def get_BBT(M,selected_row):
    R = []
    for i in range(len(M)):
        if i in selected_row:
            R.append(M[i])
    B = np.mat(np.transpose(R))
    print(np.shape(B))
    Bt = np.mat(R)
    BBT = B*Bt
    return BBT
def get_U(AAT,r):
    a,b = np.linalg.eig(AAT)
    rank_index = np.argsort(-a)

    U = []
    for i in range(r):
        index = rank_index[i]
        # print(a[index])
        # print(b[index])
        vector = b[index]
        vector = np.real(vector)
        vector = vector.tolist()
        U.append(vector[0])
    U=np.transpose(U)
    # print(U[0])    
    return np.mat(U)

def get_Vt(BBT,r):
    a,b = np.linalg.eig(BBT)
    rank_index = np.argsort(-a)
    Vt = []
    for i in range(r):
        index = rank_index[i]
        vector = b[index]
        vector = np.real(vector)
        vector = vector.tolist()
        Vt.append(vector[0])
    # print(Vt[0])
    # print(len(Vt))    
    return np.mat(Vt)


threshold = 0.01
max_iter = 120000

def cal_Z( M, U, Vt,selected_row,selected_col):
    # Z = np.ones((SIMILAR_RANK*SIMILAR_RANK))
    Z = np.eye(SIMILAR_RANK)
    print("init parameter matrix")
    print(Z)
    Z = np.reshape(Z,SIMILAR_RANK*SIMILAR_RANK)
    gradientZ = np.ones((SIMILAR_RANK,SIMILAR_RANK))
    step = 0.1
    for iteration in range(max_iter):
        # if iteration == 50:
        #     step =  0.001
        #近似矩阵与原矩阵差值
        Zm= np.reshape(Z , (SIMILAR_RANK,SIMILAR_RANK))
        C = U*Zm*Vt-M
        sample_func(C,selected_row,selected_col)
        for i in range(SIMILAR_RANK):
            for j in range(SIMILAR_RANK):
                g = calGradient(U,Vt,i,j,C)
                gradientZ[i][j] = g
        
        gradientZt = np.reshape(gradientZ,SIMILAR_RANK*SIMILAR_RANK)
        # print(str(gradientZt) + ' iteration: ' + str(iteration))
        old_z = Z
        Z = Z -gradientZt*step
        norm = np.linalg.norm(old_z - Z)
        print("norm: " + str(norm) + " iteration: " + str(iteration))

        if norm <= threshold:
            print("threshold reached")
            break
    
    Z = np.reshape(Z,(SIMILAR_RANK,SIMILAR_RANK))
    return Z


def calGradient(U,Vt,i,j,C):
    Ui = U[:,i]
    Vi = Vt[j]
    P = Ui*Vi
    
    shape = np.shape(C)
    n = shape[0]
    m = shape[1]

    P = np.reshape(P,m*n)
    C = np.reshape(C,m*n)
    return P*np.transpose(C)

def sample_func(M,selected_row,selected_col):
    shape = np.shape(M)
    n = shape[0]
    m = shape[1]
    for i in range(n):
        for j in range(m):
            if i in selected_row or j in selected_col:
                continue
            M[i,j] = 0

def sample_without_missing(M,selected_row,selected_col):
    shape = np.shape(M)
    n = shape[0]
    m = shape[1]
    mc = np.copy(M)
    for i in range(n):
        for j in range(m):
            if i in selected_row or j in selected_col:
                continue
            mc[i,j] = 0
    return mc

def sample_missing(M,selected_row,selected_col):
    shape = np.shape(M)
    n = shape[0]
    m = shape[1]
    mc = np.copy(M)
    for i in range(n):
        for j in range(m):
            if i in selected_row or j in selected_col:
                mc[i,j] = 0            
    return mc

SAVING_PATH = 'cur_result/'
MAX_RMSE = 2
#python CUR_p.py

def testCUR():
    # for iter in range(10):
        selected_row,selected_col,M = get_M()
        #M的秩为12
        # print(np.linalg.matrix_rank(M))

        # AAT = get_AAT(M,selected_col)
        # BBT = get_BBT(M , selected_row) 
        # U = get_U(AAT,SIMILAR_RANK)
        # Vt = get_Vt(BBT,SIMILAR_RANK)

        A = get_A(M,selected_col)
        B = get_B(M,selected_row)
        U = get_feature_vector(A)
        U = np.mat(np.transpose(U))
        Vt = get_feature_vector(B)
    
        # print(np.shape(U))
        # print(np.shape(V))

        Z = cal_Z(M,U,Vt,selected_row,selected_col)
        # Z = cal_Z(get_full_M(),U,Vt,selected_row,selected_col)
        similar_m = U*Z*Vt

        shape = np.shape(similar_m)
        n = shape[0]
        m = shape[1]
        row_num = len(selected_row)
        col_num = len(selected_col)
        missing_num = n*m -col_num*n - row_num*m + row_num*col_num

        print("parameter matrix")
        print(Z)

        # if result < MAX_RMSE:
        # np.save(SAVING_PATH + 'z' + str(iter) , Z )
        # np.save(SAVING_PATH + 'U' + str(iter) , U )
        # np.save(SAVING_PATH + 'Vt' + str(iter) , Vt )
        return similar_m
    

if __name__ == '__main__':
    testCUR()

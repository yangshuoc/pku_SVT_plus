import numpy as np
import time
import math
import random
from numpy import linalg as la
from sparsesvd import sparsesvd
from scipy.sparse.linalg import norm
import scipy.sparse as ss
import scipy.io
import random
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import argparse

#自定义模块
import DataFilter as df
import CUR_p as cur

def SVT1(M1,iter_num):
    
    n1,n2 = M1.shape
    total_num = len(M1.nonzero()[0])
    proportion = 1.0
    idx = random.sample(range(total_num),int(total_num*proportion))
    Omega = (M1.nonzero()[0][idx],M1.nonzero()[1][idx])
    p = 0.5
    # tau=20000
    tau = 5*math.sqrt(n1*n2)
    delta = 2
    maxiter = iter_num
    tol = 0.001
    incre = 5
    # incre = 1

    # SVT
    r = 0
    b = M1[Omega]
    P_Omega_M = ss.csr_matrix((np.ravel(b),Omega),shape = (n1,n2))
    normProjM = norm(P_Omega_M)
    k0 = np.ceil(tau / (delta*normProjM))
    Y = k0*delta*P_Omega_M
    rmse = []
    
    x = M1
    for k in range(maxiter):
        #print str(k+1) + ' iterative.'
        s = r + 1
        is_ValueError = False
        while True:
            try:
                u1,s1,v1 = sparsesvd(ss.csc_matrix(Y),s)
            except ValueError as e:
                print('ValueError:',e)
                is_ValueError = True
                break           
            # if s1[s-1] <= tau : break
            if s1[-1] <=tau:break
            s = min(s+incre,n1,n2)
            if s == min(n1,n2): break
        if is_ValueError == True:
            is_ValueError = False
            continue            
        # print(k)
        r = np.sum(s1>tau)
        U = u1.T[:,:r]
        V = v1[:r,:]
        S = s1[:r]-tau
        x = (U*S).dot(V)
        x_omega = ss.csr_matrix((x[Omega],Omega),shape = (n1,n2))                                      

        # print(k)

        if norm(x_omega-P_Omega_M)/norm(P_Omega_M) < tol:
            break

        diff = P_Omega_M-x_omega
        Y += delta*diff
        #rmse.append(norm(diff)/np.sqrt(n1*n2))
        rmse.append(la.norm(M1[M1.nonzero()]-x[M1.nonzero()]) / np.sqrt(len(x[M1.nonzero()])))
        # print(rmse[-1])

    return x,rmse
def SVT(M1,iter_num):
    # parameter initialized
    n1,n2 = np.shape(M1)
    r = np.linalg.matrix_rank(M1)
    M = np.array(M1)
    df = r*(n1+n2-r);
    oversampling = 5; 
    m = min(5*df,round(.99*n1*n2)); 
    p  = m/(n1*n2);
    ind = random.sample(range(n1*n2),m)
    Omega = np.unravel_index(ind, (n1,n2))

    data = M[Omega]

    tau = 5*math.sqrt(n1*n2); 
    delta = 2
    maxiter = iter_num
    tol = 1e-4
    incre = 5

    """
    SVT
    """
    start = time.clock()

    b = data

    r = 0
    #b是个一维数组，omega二维表示b的每个元素在矩阵中的位置。缺省值为0
    P_Omega_M = ss.csr_matrix((b,Omega),shape = (n1,n2))

    normProjM = norm(P_Omega_M)

    k0 = np.ceil(tau / (delta*normProjM))

    Y = k0*delta*P_Omega_M
    rmse = []

    for k in range(maxiter):
        s = r + 1
        while True:
            u1,s1,v1 = sparsesvd(ss.csc_matrix(Y),s)
            if s >len(s1)-1: break
            if s1[s-1] <= tau : break
            s = min(s+incre,n1,n2)
            if s == min(n1,n2): break
        
        r = np.sum(s1>tau)
        U = u1.T[:,:r]
        V = v1[:r,:]
        S = s1[:r]-tau
        x = (U*S).dot(V)
        x_omega = ss.csr_matrix((x[Omega],Omega),shape = (n1,n2))

        if norm(x_omega-P_Omega_M)/norm(P_Omega_M) < tol:
            break
        
        Y += delta*(P_Omega_M-x_omega)
        diff = ss.csr_matrix(M-x)
        rmse.append(norm(x_omega-P_Omega_M) / np.sqrt(n1*n2))
    return x,rmse



def testSVT( isMissing = True ):
    matrix = df.loadCSVData(df.DATA_FILE)
    selected_row = [1,5,4]
    selected_col = [3,4,5]

    if isMissing == True:
        selected_row,selected_col,matrix = df.buildTestMatrix(matrix)    

    row = []
    col = []
    vector,n,m = df.getCsrMatrix(matrix,row,col)
    M_original = matrix
    #增加迭代次数无法提高
    similar_m,rmse = SVT(M_original,500)
    print ('Original Matrix: ')
    # M = M_original.toarray()
    M = M_original
    print (np.array(M))
    print ('Recovered Matrix: ')
    print(similar_m)


    full_m = np.mat(cur.get_full_M())
    print("full rmse")
    #2.8749852188994565,12.98841346119567
    rmse = np.linalg.norm(similar_m - full_m)/np.sqrt(n*m)
    print(rmse)
    sample_m = cur.sample_without_missing(full_m,selected_row,selected_col)
    similar_m_sample = cur.sample_without_missing(similar_m ,selected_row,selected_col)
    
    row_num = len(selected_row)
    col_num = len(selected_col)
    missing_num = n*m -col_num*n - row_num*m + row_num*col_num
    print("without missing rmse")
    #0.8126469675019425,1.0491930175570114
    print(np.linalg.norm(similar_m_sample - sample_m)/np.sqrt(n*m-missing_num))

    sample_m2 = cur.sample_missing(full_m,selected_row,selected_col)
    similar_m_sample2 = cur.sample_missing(similar_m,selected_row,selected_col)
    # print(sample_m2)
    # print(similar_m_sample2)
    print("missing rmse")
    #6.173047284931273,28.745749834402837
    print(np.linalg.norm(similar_m_sample2 - sample_m2)/np.sqrt(missing_num))
    return similar_m,rmse

SAVING_PATH = 'cur_result/'

def testCURSVT(cur_m,rank=10):
    matrix = cur_m
    if len(matrix) == 0:
        cur.SIMILAR_RANK = rank   
        selected_row,selected_col,M = cur.get_M()
        A = cur.get_A(M,selected_col)
        B = cur.get_B(M,selected_row)
        #太慢
        U = cur.get_feature_vector(A)
        U = np.mat(np.transpose(U))
        Vt = cur.get_feature_vector(B)

        print("calculate parameters")
        Z = cur.cal_Z(cur.get_full_M(),U,Vt,selected_row,selected_col)

        # U =  np.load(SAVING_PATH + 'U'  + ".npy")
        # Vt = np.load(SAVING_PATH + 'Vt'  + ".npy")
        # Z = np.load(SAVING_PATH + 'z'  + ".npy")
        # U = np.mat(U)
        print("final parameter matrix")
        print(Z)

        matrix = U*Z*Vt
    selected_row,selected_col,M = cur.get_M()
    
    matrix = matrix.tolist()
    row = []
    col = []
    vector,n,m = df.getCsrMatrix(matrix,row,col)

    # M_original = ss.csr_matrix((vector,(row,col)),shape = (n,m))
    M_original = matrix

    similar_m,rmse = SVT(M_original,500)
    print ('Original Matrix: ')
    # M = M_original.toarray()
    M = np.array(M_original)
    print (M)
    print ('Recovered Matrix: ')
    print(similar_m)

    x_v = np.reshape(similar_m,n*m)
    M_v = np.reshape(M,n*m)
    dis = x_v-M_v

    full_m = np.mat(cur.get_full_M())
    print("full rmse")
    #17.884135360980427,17.89080305403029
    #17.88157891498195
    full_rmse = np.linalg.norm(similar_m - full_m)/np.sqrt(n*m)
    print(full_rmse)
    sample_m = cur.sample_without_missing(full_m,selected_row,selected_col)
    similar_m_sample = cur.sample_without_missing(similar_m ,selected_row,selected_col)
    
    row_num = len(selected_row)
    col_num = len(selected_col)
    missing_num = n*m -col_num*n - row_num*m + row_num*col_num
    print("without missing rmse")
    #11.024466797398045,19.874140345126662
    #20.03008114246185
    print(np.linalg.norm(similar_m_sample - sample_m)/np.sqrt(n*m-missing_num))

    sample_m2 = cur.sample_missing(full_m,selected_row,selected_col)
    similar_m_sample2 = cur.sample_missing(similar_m,selected_row,selected_col)
    # print(sample_m2)
    # print(similar_m_sample2)
    print("missing rmse")
    #norm 136.0353367143714
    #33.135220249606,5.117295490779692
    #0.38141366418824263
    print(np.linalg.norm(similar_m_sample2 - sample_m2)/np.sqrt(missing_num))
    return similar_m,full_rmse


if __name__ == '__main__':
    msg = """
            Usage:
            SVT: 
                python SVT.py --mode SVT
            SVT plus:
                python SVT.py --mode SVTp  --rank 10
            SVT without missing: 
                python SVT.py --mode SVT --missing false
            """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='SVT',
                        help=u'使用SVT算法')
    parser.add_argument('--missing', type=str, default='true',
                        help='是否缺失')
    parser.add_argument('--rank', type=int, default=10,
                        help='映射矩阵的秩')
    args = parser.parse_args()

    if args.mode == 'SVT':
        if args.missing == 'false':
            testSVT(False)
        else:
            testSVT()        
    elif args.mode == 'SVTp':
        testCURSVT([],args.rank)
    else:
        print(msg)
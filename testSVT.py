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
import csv
import DataFilter as df

# parameter initialized
n1, n2, r = 150, 300, 10
#150*10乘10*300
left = np.random.random((n1,r))
right = np.random.random((r,n2))
M = np.random.random((n1,r)).dot(np.random.random((r,n2)))
# M = np.array(df.loadCSVData(df.DATA_FILE))
df = r*(n1+n2-r);
oversampling = 5; 
m = min(5*df,round(.99*n1*n2)); 
p  = m/(n1*n2);
ind = random.sample(range(n1*n2),m)
Omega = np.unravel_index(ind, (n1,n2))

data = M[Omega]

tau = 5*math.sqrt(n1*n2); 
delta = 2
maxiter = 400
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
    
print ('calculating time: ' + str(time.clock() - start))
print ('Recovered Matrix: ')
print (x)

CSV_FILE = 'svt_matrix.csv'
def saveCSV(matrix):
    writer = csv.writer(open(CSV_FILE, mode='w',encoding='utf-8',newline=''))
    writer.writerow(matrix[0])
    for m in matrix:
        writer.writerow(m)
print ('Original Matrix: ')
print (M)
print(np.linalg.matrix_rank(M))
saveCSV(M)


x_v = np.reshape(x,45000)
M_v = np.reshape(M,45000)
dis = x_v-M_v
# print(dis)
print("rmse:")
print(np.linalg.norm(dis)/np.sqrt(n1*n2))

x_coordinate = range(len(rmse))
plt.ylim(0,0.5)
plt.xlabel('Number of iterations')
plt.ylabel('RMSE')
plt.plot(x_coordinate,rmse,'-')
plt.show()
print(np.linalg.matrix_rank(M))
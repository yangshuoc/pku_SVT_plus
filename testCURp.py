from numpy import *

def calGradient(U,Vt,i,j,C):
    Ui = U[:,i]
    Vi = Vt[j]
    P = Ui*Vi
    P = reshape(P,9)
    C = reshape(C,9)
    return P*transpose(C)

# M=  [[5,4,1,1,3,1],
#     [1,3,5,3,1,1],
#     [2,1,4,5,1,1],
#     [2,1,1,2,5,3],
#     [1,2,5,3,3,5]]

M=  [[5,100,1],
     [1,3,5],
     [2,1,4]]


M = mat(M)

#列
# C= [[5,1,2],[1,5,4] ]
C= [[5,1,2,2,1],[1,5,4,1,5] ]
# n*d
A = mat(transpose(C))
At = mat(C)
AAT = A*At
print(AAT)

R =[[1,3,5,3,1,1],[2,1,4,5,1,1],[1,2,5,3,3,5]]
#m*d
B = mat(transpose(R))
Bt = mat(R)
BBT = B*Bt
print (BBT)


#特征值,特征向量
a,b = linalg.eig(AAT)
# print(a)
# print(b)

#特征值是复数????
a,b = linalg.eig(BBT)
# print(a)
# print(b)

#r = 2
#U [-0.36011056 -0.84644801 -0.39223227 -0.25856307 -0.01387933]
# [-0.54035374  0.32076414 -0.19611614 -0.55001326 -0.21162489]
#V [[ 0.13342623+0.j          0.98896735+0.j          0.06435071+0.j 
# 0.96886665+0.j          0.27352486+0.15592897j  0.27352486-0.15592897j]
# [ 0.32454358+0.j         -0.06973488+0.j          0.39879729+0.j
# -0.03908775+0.j         -0.17980418+0.38691406j -0.17980418-0.38691406j]
#特征向量
U = transpose([[ 0.50372007,0.84131121,-0.19611614],
                [ 0.62593155,-0.5119139 , -0.58834841]])
# U = transpose([[5.0,1.0,2.0],[1.0,5.0,4.0]])
U = mat(U)
# U = U*(1/18)
# print(U[:,1])
#复数跟实数香草之后虚部直接消失了
Vt =  [[-0.27299273,-0.69283506,-0.66742381],
       [-0.40765381 ,0.7117202 ,-0.57207755]]

# Vt =  [[1.0,3.0,5.0],[2.0,1.0,4.0]]
Vt = mat(Vt)
# Vt = Vt*(1/18)

# print(Vt[1])


Z = ones(4)
gradientZ = ones((2,2))

step = 0.001
threshold = 0.001
# threshold = 0.0001

#5000,step 0.001,threshold = 0.0001 ->6.855219254745567 可以了
#[ 0.09599833  0.01761302 -0.015548   -0.01465003] iteration4348
#100000  step = 0.001  threshold = 0.000001  6.85449988699037
#迭代到10万次梯度就不动了,很神奇
#
for iteration in range(120000):
# iteration = 0
# while True:
#     iteration+=1

    if iteration == 100000:
        step /=100
    Zm= reshape(Z , (2,2))

    C = U*Zm*Vt-M
    C[0,1] = 0
    for i in range(2):
        for j in range(2):
            g = calGradient(U,Vt,i,j,C)
            gradientZ[i][j] = g
    Z = reshape(Z,4)
    gradientZt = reshape(gradientZ,4)
    print(str(gradientZt) + ' iteration: ' + str(iteration))
    old_z = Z
    Z = Z -gradientZt*step
    norm = linalg.norm(old_z - Z)
    print("norm: " + str(norm) + " iteration: " + str(iteration))
    if norm <= threshold:
        print("threshold reached")
        break

Z = reshape(Z,(2,2))
result = U*Z*Vt
print(result)
result[0,1] = 0
M[0,1] = 0
#不清零96.95053083599366
#清零6.096136914545958
print(linalg.norm(result - M))

# temp = copy(M)
# temp = mat(temp)

# temp[0,1] = result[0,1]
# print(linalg.norm(temp - M))


#梯度爆炸
# [4.51903861e+307 3.42751182e+307 6.26983114e+307 4.75541863e+307]
# [-8.88651780e+307 -6.74007184e+307 -1.23293848e+308 -9.35135015e+307]
# [1.74749998e+308 1.32540953e+308             inf             inf]
# [-inf -inf -inf -inf]
# d:\pku_workspace\CUR\testCURp.py:88: RuntimeWarning: invalid value encountered in subtract
#   Z = Z -gradientZt*step
# [nan nan nan nan]

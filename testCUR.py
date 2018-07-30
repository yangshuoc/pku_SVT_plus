from numpy import *

# A=  [[5,200,1],
#     [1,3,5],
#     [2,1,4]]

A=  [[5,4,1,1,3,1],
    [1,3,5,3,1,1],
    [2,1,4,5,1,1],
    [2,1,1,2,5,3],
    [1,2,5,3,3,5]]


aMat = mat(A)

C= [[5,1,2,2,1],[1,5,4,1,5] ]
cMat = mat(transpose(C))
# print(cMat)

R =[[1,3,5,3,1,1],[2,1,4,5,1,1]]
rMat = mat(R)

Cp=linalg.pinv(cMat)
# print(C1)
Rp=linalg.pinv(rMat)
# print(Rp)

U=Cp*aMat*Rp
# print(U)

# U=mat([[1,1],[3,3],[1,5]])

result = cMat*U*rMat
print(result)

#清零
result[0,1] = 0
A[0][1] = 0

print(linalg.norm(result - A))
#清零97.97931052382067，不清零更大
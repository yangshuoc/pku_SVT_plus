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
import DataFilter as df

SIMILAR_RANK = 50

def get_full_M():
    return df.loadCSVData(df.DATA_FILE)

SAVING_PATH = 'cur_result/'

if __name__ == '__main__':
    Y = get_full_M()
    ut, s, vt = sparsesvd(ss.csc_matrix(Y),SIMILAR_RANK)
    ut = np.mat(np.transpose(ut))
    z = np.mat(np.diag(s))
    vt = np.mat(vt)
    result = ut*z*vt

    print("result rmse: ")
    print(np.shape(z))
    print(np.linalg.norm(result - Y)/np.sqrt(150*300))

    np.save(SAVING_PATH + 'z' , z )
    np.save(SAVING_PATH + 'U' , ut)
    np.save(SAVING_PATH + 'Vt' , vt )
    
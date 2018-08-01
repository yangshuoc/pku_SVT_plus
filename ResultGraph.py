import SVT
import matplotlib.pyplot as plt
import os
import math
def repeatTest(num,rank):
    svt_rmse = []
    svtp_rmse = []
    svt_nomissing_rmse = []
    for i in range(num):
        svt_m,svtRmse = SVT.testSVT()
        svt_rmse.append(svtRmse)
        svt_m,svtpRmse = SVT.testCURSVT([],rank)
        svtp_rmse.append(svtpRmse)
        # svt_m,SVTNoRmse = SVT.testSVT(False)
        # svt_nomissing_rmse.append(SVTNoRmse)
        svt_nomissing_rmse.append(0.005547751681770088)
        os.system('cls')
        print(i)
    return svt_rmse,svtp_rmse,svt_nomissing_rmse
def plotRepeatGraph():
    svt_rmse,svtp_rmse,svt_nomissing_rmse = repeatTest(10,5)
    # svt_rmse = [1.6,1.5,1.6,1.5,1.6,1.5,1.6,1.5,1.6,1.5]
    # svtp_rmse = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    # svt_nomissing_rmse = [0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002]
    x_coordinate = range(len(svt_rmse))
    plt.xlabel('experiment times')
    plt.ylabel('RMSE')
    plt.plot(x_coordinate,svt_rmse,color='black',linewidth=1.0, linestyle='-')
    plt.plot(x_coordinate, svtp_rmse, color='red', linewidth=1.0, linestyle='-')
    plt.plot(x_coordinate, svt_nomissing_rmse, color='blue', linewidth=1.0, linestyle='--')
    plt.show()

MAX_RANK = 30
def plotRanksGraph():
    svtp_rmse = []
    svt_nomissing_rmse = []
    for rank in range(1,MAX_RANK):
        svt_m,svtpRmse = SVT.testCURSVT([],rank)
        svtp_rmse.append(svtpRmse)
        # svt_m,SVTNoRmse = SVT.testSVT(False)
        # svt_nomissing_rmse.append(SVTNoRmse)
        svt_nomissing_rmse.append(0.005547751681770088)
        os.system('cls')
        print(rank)
    x_coordinate = range(1,MAX_RANK)
    plt.xlabel('RANK')
    plt.ylabel('RMSE')
    plt.plot(x_coordinate, svtp_rmse, color='red', linewidth=1.0, linestyle='-')
    plt.plot(x_coordinate, svt_nomissing_rmse, color='blue', linewidth=1.0, linestyle='--')
    plt.show()


def plotMissingProportionGraph():
    total = 90*522
    prop = 0.1
    svt_rmse_avg_list = []
    svtp_rmse_avg_list = []
    svt_nomissing_rmse_avg_list=[]
    x_coordinate = []
    while prop < 0.95:
        print("Proportion")
        print(prop)
        x_coordinate.append(prop)
        missing = total*prop
        missing_col_num = int(math.sqrt((missing/6)))
        missing_row_num = int(6*math.sqrt((missing/6)))
        SVT.df.MISSING_COL_NUM = missing_col_num
        SVT.df.MISSING_ROW_NUM = missing_row_num
        svt_rmse,svtp_rmse,svt_nomissing_rmse = repeatTest(3,5)
        svt_rmse_avg = sum(svt_rmse)/len(svt_rmse)
        svt_rmse_avg_list.append(svt_rmse_avg)
        svtp_rmse_avg = sum(svtp_rmse)/len(svtp_rmse)
        svtp_rmse_avg_list.append(svtp_rmse_avg)
        svt_nomissing_rmse_avg =  0.005547751681770088
        svt_nomissing_rmse_avg_list.append(svt_nomissing_rmse_avg)
        prop = prop + 0.1
        
    

    plt.xlabel('Missing Proportion')
    plt.ylabel('rmse')
    plt.plot(x_coordinate,svt_rmse_avg_list,color='black',linewidth=1.0, linestyle='-')
    plt.plot(x_coordinate, svtp_rmse_avg_list, color='red', linewidth=1.0, linestyle='-')
    plt.plot(x_coordinate, svt_nomissing_rmse_avg_list, color='blue', linewidth=1.0, linestyle='--')
    plt.show()

if __name__ == '__main__':
    # plotRepeatGraph()
    # plotRanksGraph()
    plotMissingProportionGraph()



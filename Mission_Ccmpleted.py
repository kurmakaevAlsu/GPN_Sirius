from numpy import array, append, zeros, squeeze, delete, concatenate, hstack, vstack, ones, linspace, nan_to_num, nan, savetxt
import numpy as np
from re import sub
from scipy.interpolate import interpn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
import time
import sys
import argparse

def CreateParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument('--THP', type = float, default = 7)
    parser.add_argument('--WCT', type = float, default = 0.57)
    parser.add_argument('--GOR', type = float, default = 97.)
    parser.add_argument('--iter', type = int, default = 100)
    parser.add_argument('--pres', type = int, default = 181)
    parser.add_argument('--PI_1', type = float, default = 9)
    parser.add_argument('--PI_2', type = float, default = 8)
    return parser
class clVfp:
    def __init__(self,includeFile,Write_X,Write_Y):
        """
            currentIncludeFile - stored file
            varsValue, varsName - values of variables and its names
            data - multidimension array of bhp-values
            zeroDimFilter - filter by zero-dimension variables
        """
        self.currentIncludeFile = includeFile
        self.write_x=Write_X
        self.write_y=Write_Y
        self.varsValue,self.varsName,self.data,self.zeroDimFilter = self.readVfp(includeFile)

    def calc(self,rate,thp,wfac,gfac,alq):
        params = array([alq,gfac,wfac,thp,rate])[self.zeroDimFilter]
        return interpn(self.varsValue,self.data,params,method='linear',bounds_error=False,fill_value=None)[0]

    def readVfp(self,includeFile):

        def handlingLine(line):
            slash_pos=line.find("/")
            if slash_pos>0:
                line=line[:slash_pos+1]
            if "--" in line:
                line=sub("--.*","",line)
            return line.rstrip()

        def convertToNumber(arr):
            result=[]
            for value in arr:
                if "*" in value:
                    value = value.replace(" ","")
                    n, value = value.split("*")
                    n=int(n)
                    value=float(value)
                    for i in range(n):
                        result.append(value)
                else:
                    result.append(float(value))
            return array(result)

        # -------------------------------------------------------------------------- #
        # Output variables:
        #  varsName = array(['LIQ','WCT',...,'ALQ'])
        #  varsValue = array([
        #                   [var00,var01,...]   - 'LIQ'
        #                   [var10,var11,...]   - 'THP'
        #                    ...
        #               ])
        #
        #  data = array[...]['THP']['LIQ']
        #

        headerOrder=[0,3,1,2,4]
        varsName=[]
        varsValue=[[]]
        data=[]
        varsDimension=[]



        # Help variables:
        dataSection=False
        nextRecord = True
        header=0
        varsNumbersLine=array([])
        dataLine=array([])

        # read vfp tables
        with open(includeFile) as f:
            for line in f:
                line=handlingLine(line)

                if "VFPPROD" in line:
                    dataSection=True
                    header=1
                    continue
                if not line:
                    continue


                if dataSection:
                    line_splited=array(line.split())

                    # Reading first record of keyword
                    # Remember names of variables
                    if header==1:
                        varsName = line_splited[2:6]
                        varsName = append(varsName,"ALQ")
                        varsValue = [[] for i in range(5)]
                        header+=1
                        continue

                    # Reading from 2th to 6th records of keyword
                    # Remember values of variables
                    elif header>1 and header<7:
                        varsValue[header-2] = append(varsValue[header-2],line_splited)
                        if line[-1]=="/":
                            varsValue[header-2]=varsValue[header-2][:-1].astype(float)
                            varsDimension=append(varsDimension,len(varsValue[header-2]))
                            header+=1
                        continue

                    # Preparing output array
                    elif header==7:
                        data=zeros(tuple(varsDimension[::-1].astype(int)))
                        nextRecord = True
                        header+=1

                    # Reading main section of keyword
                    # Remember bhp values
                    if nextRecord:
                        # Remember number of variables
                        varsNumbersLine = line_splited[0:4].astype(int)
                        line_splited=line_splited[4:]
                        dataLine = []
                        nextRecord = False

                    dataLine = append(dataLine,line_splited)
                    if line[-1]=="/":
                        dataLine=convertToNumber(dataLine[:-1])
                        data[varsNumbersLine[3]-1,varsNumbersLine[2]-1,varsNumbersLine[1]-1,varsNumbersLine[0]-1,:]=dataLine
                        nextRecord = True

        # skip zero dimensions
        # reoedering arrays
        s=array(data.shape)[::-1]
        varsValue=array(varsValue)
        varsName=array(varsName)
        X_train=[]
        Y_train=[]      
        
        count=varsDimension[0].astype(int)*varsDimension[1].astype(int)*varsDimension[2].astype(int)*varsDimension[3].astype(int)*varsDimension[4].astype(int)
        X_train=np.zeros((count,4))
        Y_train=np.zeros(count)
        ML=zeros((count,5))
        k=0
        while (k!=count):
            for j in range(varsDimension[1].astype(int)): #идем по значениям THP
                for n in range(varsDimension[2].astype(int)): #идем по значениям WCT
                    for m in range(varsDimension[3].astype(int)): #идем по значениям GOR
                        for i in range(varsDimension[0].astype(int)): #идем по значениям LIQ
                            Y_train[k]=data[0][m][n][j][i]
                            X_train[k][0]=varsValue[0][i]
                            X_train[k][1]=varsValue[1][j]
                            X_train[k][2]=varsValue[2][n]
                            X_train[k][3]=varsValue[3][m]
                            k+=1
        np.save(self.write_x, X_train)
        np.save(self.write_y, Y_train)
        return varsValue[s!=1][::-1],varsName[headerOrder][s!=1][::-1],squeeze(data),array(s!=1)[::-1]
#  realization of LinearRegression

def LinRegression(X_train,Y_train):
    model = LinearRegression().fit(X_train, Y_train)
    r_sq = model.score(X_train, Y_train)
    #print('absolute_error =', mean_absolute_error((np.dot(X_train, model.coef_)+model.intercept_),Y_train))
    return(model)
#  realization of LinearRegression

def CatBoost(X_train,Y_train):
    model = CatBoostRegressor().fit(X_train, Y_train)
    r_sq = model.score(X_train, Y_train)
    #print('absolute_error =', mean_absolute_error((np.dot(X_train, model.coef_)+model.intercept_),Y_train))
    #return('coefficient of determination:', r_sq, 'model.coef:', model.coef_, 'model.intercept:', model.intercept_, 'absolute_error:', absolute_error)
    return(model)
def Find_Pz_from_LinRegressio(liq, thp, wct, gor, Model):
    p = ((liq*Model.coef_[0]) + (thp*Model.coef_[1]) + (wct*Model.coef_[2])+(gor*Model.coef_[3])+Model.intercept_)
    return(p)
def IPR(Q, PI, P_pres):
    return(P_pres - Q/PI)
X=[]
Y=[]
parser=CreateParser()
args=parser.parse_args() 
THP_inlet = args.THP
WCT =args.WCT
GOR = args.GOR
P_pres = args.pres
iter_count=args.iter
PI_1 = args.PI_1
PI_2 = args.PI_2

#Парсинг файлов VFP
Well_All = clVfp("VFP_WELL_ALL.txt","Well_ALL_X.npy",'Well_ALL_Y.npy')
Pipes_All = clVfp("VFP_PIPES_ALL.txt","Pipes_ALL_X.npy","Pipes_ALL_Y.npy")
Well_THP = clVfp("VFP_WELL_THP.txt","Well_THP_X.npy","Well_THP_Y.npy")
start_time = time.time()

X = np.load('Well_ALL_X.npy')
Y = np.load('Well_ALL_Y.npy')
Well_ALL_regression = LinRegression(X,Y)
#Well_ALL_regression = CatBoost(X,Y)

X = np.load('Pipes_ALL_X.npy')
Y = np.load('Pipes_ALL_Y.npy')
Pipes_ALL_regression = LinRegression(X,Y)

X = np.load('Well_THP_X.npy')
Y = np.load('Well_THP_Y.npy')
Well_THP_regression = LinRegression(X,Y)

Matrix_of_scheme = np.load('System.npy')
n = np.shape(Matrix_of_scheme)[0]
# WCT = float(input('Введите обводнённость (%):')) 
# GOR = float(input('Введите газовый фактор (sm3/sm3):'))  
# THP_inlet = float(input('Введите начальное устьевое давление (barsa):'))  



Qmin_0 = 0
Qmin_1 = 0
Qmax_0 = PI_1*(P_pres-1)
Qmax_1 = PI_2*(P_pres-1)
steps=0
flag=True
while flag:
    steps+=1
    Q = zeros(n)
    P = zeros(n)
    P[n-1]=THP_inlet
    mid_0= Qmin_0+(Qmax_0-Qmin_0)/ 2
    mid_1= Qmin_1+(Qmax_1-Qmin_1) / 2
    Q[0] = mid_0
    Q[1] = mid_1
    P_ipr_1 = IPR(Q[0], PI_1, P_pres)
    P_ipr_2 = IPR(Q[1], PI_2, P_pres)
    eps_1=P_ipr_1*0.01
    eps_2=P_ipr_2*0.01
    for i in range(n):
        for j in range(n):
            if Matrix_of_scheme[i][j] != 0:
                Q[j] += Q[i]
    for i in range(1,n+1):
        for j in range(1,n+1):
            if Matrix_of_scheme[n-i][n-j]==0:
                continue
            elif Matrix_of_scheme[n-i][n-j]==1:
                P[n-i]=Find_Pz_from_LinRegressio(Q[n-i], P[n-j], WCT, GOR, Well_ALL_regression)
                #P[n-i]=Well_ALL_regression.predict([Q[n-i], P[n-j], WCT, GOR])#
            elif Matrix_of_scheme[n-i][n-j]==2:
                P[n-i]=Find_Pz_from_LinRegressio(Q[n-i], P[n-j], WCT, GOR, Well_THP_regression)
            elif Matrix_of_scheme[n-i][n-j]==3:
                P[n-i]=Find_Pz_from_LinRegressio(Q[n-i], P[n-j], WCT, GOR, Pipes_ALL_regression)
    if ((abs(P[0]-P_ipr_1)>eps_1 or abs(P[1]-P_ipr_2)>eps_2) and steps<=iter_count):
        if P[0]<(P_ipr_1-eps_1):
            Qmin_0=Qmin_0+mid_0
        elif P[0]>(P_ipr_1+eps_1):
            Qmax_0=Qmax_0-mid_0
        if P[1]<(P_ipr_2-eps_2):
            Qmin_1=Qmin_1+mid_1
        elif P[1]>(P_ipr_2+eps_2): 
            Qmax_1=Qmax_1-mid_1
    else: 
        flag=False
end_time=time.time()
m=len(P)
print('        Давление      Дебит')
for i in range(1,m+1):
    print("{:5d}".format(m-i),"{:>10.3f}".format(P[m-i]),"{:>10.3f}".format(Q[m-i]))
print('Время работы программы: ',end_time-start_time)
print('Число итераций: ',steps)
print('Точность для скважины 1 (%): ', abs(P_ipr_1-P[0])/P_ipr_1*100)
print('Точность для скважины 2 (%): ', abs(P_ipr_2-P[1])/P_ipr_2*100)
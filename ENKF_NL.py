# -*- coding: utf-8 -*-

from copy import deepcopy


import numpy as np
from numpy import zeros
from numpy.random import multivariate_normal
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from numpy.random import randn
from numpy import linalg as LA

with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0

'''


### Evaluation




'''
def error(gt,b,M):
    error=np.sqrt(np.sum((gt-b)**2,axis=0))
    return((1/M)*error)


# Vorwärstsschritt und Berechnung der Kovarianzmatrix P
def Forward_step(X_n_minus_one,h,E,std):


    # Create M Forward Samplex
    
    
    Sample_X=np.zeros(h)
    
    
    # C identity
    C=np.eye(h)
    
    W=np.random.normal(0,0.3,M)
    #print(x_i_minus_one)
    
    for i in range(0,h):
        
        # apply euler approximation
        #f = lambda t, s: 2*t-4*t**3 
        #Sample_X[i]= x_i_minus_one[i]+Schritt*(2*x_i_minus_one[i]-4*x_i_minus_one[i]**3) 
        #Sample_X[i]= x_i_minus_one[i]+Schritt*(-np.sin(x_i_minus_one[i])) 
        Sample_X[i]=  np.cos(X_n_minus_one[i])
        
    forward_sum=(np.sum(Sample_X)) 
    #Sample_X= Sample_X + np.matmul(C,W)
    Sample_X= Sample_X + W
    # forward mean
    forward_sum=(1/h)*(forward_sum)+np.mean(W)
    for k in range(0,h):
        E[k]= Sample_X[k]-forward_sum
        
    
    P= 1/(h-1)*np.matmul(E.T,E)
    
    
    return [Sample_X,P,E]
    
def K_(P,std): # function for calculating the Kalman gain 
        
        # all 1D
        H=1
        # Standard deviation in R
        # No concrete guidelines found
        # derived after testing
        R=std**2
        
        K= P*H*(1/(H*P*H+R))

   
        return K


               


def T_(P_f,E,std):# function for calculating the transformation matrix

        # H idenity matrix
        H=1
        # Standard deviation in R
        # No concrete guidelines found
        # derived after testing
        R=std**2
        I=1
        # 42,43
        
        T= I + np.matmul(E.T,H*(1/R)*H*E)
        
        T=1/np.sqrt(T)
        
        
        return T
    
def Mean_Samples(h,m_bar_minus_eins):
    m_bar=np.zeros(h)
    
    
    for i in range(0,h):
        
        # apply euler approximation
        #f = lambda t, s: 2*t-4*t**3 
        m_bar[i]= np.cos(m_bar_minus_eins[i])
    
    return m_bar

def Analysis_Step(X_n,P,h,E,std,m_bar_minus_eins):
    
    # Noise for observation
    V=np.random.normal(0,0.5,M)
    
    
    m_i_n= Mean_Samples(h,m_bar_minus_eins)

    # Scalar since in 1D
    KP=K_(P,std)   
    TP=T_(P,E,std)
    
    # H in 1D
    H=1
    gamma=1
    Y=H*X_n+gamma*V

    # Eq. 77&78
    m_bar=m_i_n+ KP*(Y-H*m_i_n)
    X_bar=m_bar+TP*(X_n-m_i_n)
    
    
    return [X_bar,Y,m_bar]
  

# M Samples, N Length
M=[1000]
std=0.6
N=50
h=1/N
# gitter
#t = np.arange(0, 1 + h, h) 
#Schritt=t[1]


# draw M samples from gt
gt=np.random.normal(0,std,100000)

#gt=gt+randn()*std #adding random noise to ground truth values

## M samples beim Zeitpunkt n
## je größer desto kleiner der Fehler
Step_Vector=0

gt=np.random.choice(gt,M)

tests=100
summe=0


for j in range(0,tests):
    for h in M:
        #Filter_Vector=np.zeros(N)
        #Filter_Vector[0]=y0
        #Y_Vector=np.zeros(N)
        X_n_minus_one=gt
        m_bar_minus_eins= np.zeros(h)
        
        E=np.zeros([h])
        
        # n Zeitschritte
        
        
        #Testwert=np.zeros(n)
        
        # approximation der posterior verteilung durch entnahme von samples
        # forward step X mit M Samples
    
        [X_n,P,E]=Forward_step(X_n_minus_one,h,E,std)
        

        
        [X_bar,Y,m_bar]=Analysis_Step(X_n,P,h,E,std,m_bar_minus_eins)
        
        m_bar_minus_eins=m_bar
        X_n_minus_one=X_bar
        X_bar_minus_eins=X_bar
        
        
        #Filter_Vector[i]=np.mean(X_bar)
        #Y_Vector[i]=np.mean(Y)
        #gt=np.random.normal(0,std,M)
        summe+=error(gt,X_bar,M[0])
        
        
        #summe=(1/M[0])*(np.sum((abs(np.random.normal(0,std,M)-X_bar))))
        
    

print(summe/tests)
# # # DGL
# f = lambda t, s: 2*t-4*t**3 
# # # step

# h=1/N
# # # gitter
# t = np.arange(0, 1 + h, h) 
# # # AB y0=0
# y0 = 0

# Init=np.zeros(N)

# #  #   for i in range(0,M2):
# Init[0]=y0
# # #print(Init[0])
# for i in range(1,N):
#      Init[i]= Init[i-1]+ h*f(t[i], Init[i])
# #np.savetxt("ground_truth_dgl.txt",Init)
# #y=np.loadtxt("ground_truth_dgl.txt")

# #plt.plot(t[0:N],y,label = "Solution") 
# plt.plot(t[0:N],Filter_Vector,label = "ENKF") 
# plt.plot(t[0:N],Y_Vector,label = "Observation") 
# plt.legend(loc='best')
# plt.title("Ensemble Kalman Filter with 100 Samples")
# plt.show()

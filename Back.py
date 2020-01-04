
from scipy.io import loadmat
import numpy as np
import pandas as pd
data = loadmat('ex4data1.mat')
X = data['X']
y = data['y']
print(X)

for i in range(5000):
    if y[i]==10:
        y[i]=0
print(y)

uni = np.unique(y)
print(uni)

weights = loadmat('ex4weights.mat')
weights.keys()

print('X: {} (with intercept)'.format(X.shape))
print('y: {}'.format(y.shape))

theta1, theta2 = weights['Theta1'], weights['Theta2']

print('theta1: {}'.format(theta1.shape))
print('theta2: {}'.format(theta2.shape))


wts = np.r_[theta1.ravel(), theta2.ravel()]

def sig(z):
    return 1/(1+np.exp(-z))

def gdash(z):
    return sig(z)*(1-sig(z))

print(X.shape[0])

def nnCostFunction(wts,ipsize,hide,opsize,X,y,reg):
    
    theta1 = wts[0:(hide*(ipsize+1))].reshape(hide,(ipsize+1))
    theta2 = wts[(hide*(ipsize+1)):].reshape(opsize,(hide+1))
    
    m = 5000
    y_matrix = pd.get_dummies(y.ravel()).as_matrix()
    bias = np.ones((5000,1))
    X_with_bias = np.hstack((bias,X))
    
    a1 = X_with_bias
    print(a1.shape,a1)
    z2 = theta1.dot(a1.T)
    a2 = np.c_[bias,sig(z2).T]
    z3 = theta2.dot(a2.T)                                      
    a3 = sig(z3)
    
    
    J = -1*(1/m)*np.sum((np.log(a3.T)*(y_matrix)+np.log(1-a3).T*(1-y_matrix))) + \
        (reg/(2*m))*(np.sum(np.square(theta1[:,1:])) + np.sum(np.square(theta2[:,1:])))

    d3 = a3.T - y_matrix
    d2 = theta2[:,1:].T.dot(d3.T)*gdash(z2)

    D1 = d2.dot(a1) 
    D2 = d3.T.dot(a2)
    
    t1 = np.c_[np.ones((theta1.shape[0],1)),theta1[:,1:]]
    t2 = np.c_[np.ones((theta2.shape[0],1)),theta2[:,1:]]
    
    t1_grad = D1/m + (t1*reg)/m
    t2_grad = D2/m + (t2*reg)/m
    
    return(J, t1_grad, t2_grad)

J1,_,_=nnCostFunction(wts,400,25,10,X,y,0)
J2,_,_=nnCostFunction(wts,400,25,10,X,y,1)    

print(nnCostFunction(wts,400,25,10,X,y,0)[0])
print(J2)
    
    
    






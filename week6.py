
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd
from scipy.io import loadmat

from scipy.optimize import minimize

data = loadmat('ex5data1.mat')
print(data)

y_train = data['y']
X_train = np.c_[np.ones_like(data['X']), data['X']]

yval = data['yval']
Xval = np.c_[np.ones_like(data['Xval']), data['Xval']]


def computeCost(theta, X, y, reg):
    m = y.size
    h = X.dot(theta)
    J = (1/(2*m))*np.sum(np.square(h-y)) + (reg/(2*m))*np.sum(np.square(theta[1:]))
    return(J)


def grad(theta, X, y, reg):
    m = y.size
    h = X.dot(theta.reshape(-1,1))
    gradi = (1/m)*(X.T.dot(h-y))+ (reg/m)*np.r_[[[0]],theta[1:].reshape(-1,1)]
        
    return(gradi.flatten())
    
initial_theta = np.ones((X_train.shape[1],1))
cost = computeCost(initial_theta, X_train, y_train, 0)
gradient = grad(initial_theta, X_train, y_train, 0)

print(cost)
print(gradient)

def train(X, y, reg):

    initial_theta = np.array([[15],[15]])
    
        
    res = minimize(computeCost, initial_theta, args=(X,y,reg), method=None, jac=lrgradientReg,
                   options={'maxiter':5000})
    
    return(res)
fit = train(X_train, y_train, 0)
print(fit)


def learningCurve(X, y, Xval, yval, reg):
    m = y.size
    
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))
    
    for i in np.arange(m):
        res = train(X[:i+1], y[:i+1], reg)
        error_train[i] = computeCost(res.x, X[:i+1], y[:i+1], reg)
        error_val[i] = computeCost(res.x, Xval, yval, reg)
    
    return(error_train, error_val)
t_error, v_error = learningCurve(X_train, y_train, Xval, yval, 0)
plt.plot(np.arange(1,13), t_error, label='Training error')
plt.plot(np.arange(1,13), v_error, label='Validation error')
plt.title('Learning curve for linear regression')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend();


    


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize #fmin_cg to train the linear regression
from scipy.io import loadmat
from sklearn.svm import SVC

data = loadmat('ex6data1.mat')
X = data['X']
y = data['y']

def plotData(X, y):
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    
    plt.scatter(X[pos,0], X[pos,1], s=60, c='k', marker='x', linewidths=1)
    plt.scatter(X[neg,0], X[neg,1], s=60, c='y', marker='+', linewidths=1)

plotData(X,y)

def gaussianKernel(A, B, sig=2):
    n = (A-B).T.dot(A-B)
    return(np.exp(-n/(2*sig**2)))
    
A = np.array([1, 2, 1])
B = np.array([0, 4, -1])
print(A,B)
sigma = 2

print(gaussianKernel(A, B, sigma))

def plot_svc(svc, X, y, h=0.02, pad=0.25):
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    plotData(X, y)
    
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker='|', s=100, linewidths='1')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of support vectors: ', svc.support_.size)
    
c = SVC(C=1.0, kernel='linear')
c.fit(X, y.ravel())
plot_svc(c, X, y)

c.set_params(C=1000)
c.fit(X, y.ravel())
plot_svc(c, X, y)
    
data1 = loadmat('ex6data2.mat')
data1.keys()

y1 = data1['y']
X1 = data1['X']

plotData(X1, y1)

c2 = SVC(C=50, kernel='rbf', gamma=6)
c2.fit(X1, y1.ravel())
plot_svc(c2, X1, y1)

data2 = loadmat('ex6data3.mat')
data2.keys()

y2 = data2['y']
X2 = data2['X']

plotData(X2, y2)

c3 = SVC(C=50, kernel='rbf', gamma=6)
c3.fit(X2, y2.ravel())
plot_svc(c3, X2, y2)





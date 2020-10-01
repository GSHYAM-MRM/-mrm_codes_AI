from scipy.io import loadmat

import numpy as np
from matplotlib import pyplot as plt


from sklearn.preprocessing import StandardScaler
from scipy import linalg

data3 = loadmat('ex7faces.mat')
data3.keys()

X2 = data3['X']
print('X2:', X2.shape)

scaler = StandardScaler()
scaler.fit(X2)

U, S, V = linalg.svd(scaler.transform(X2).T)
print(U)
print(S)

plt.scatter(X2[:,0], X2[:,1], s=30, edgecolors='b',facecolors='None', linewidth=1);

plt.gca().set_aspect('equal')
plt.quiver(scaler.mean_[0], scaler.mean_[1], U[0,0], U[0,1], scale=S[1], color='r')
plt.quiver(scaler.mean_[0], scaler.mean_[1], U[1,0], U[1,1], scale=S[0], color='r');








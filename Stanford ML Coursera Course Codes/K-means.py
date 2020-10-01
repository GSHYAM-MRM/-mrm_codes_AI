

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


data = loadmat('ex7data2.mat')
print(data)
X = data['X']


from sklearn.cluster import KMeans

centroids = KMeans(n_clusters=3,init = 'k-means++').fit(X)
print(centroids.cluster_centers_)

plt.scatter(X[:,0], X[:,1], s=40, c=centroids.labels_, cmap=plt.cm.prism) 
plt.title('K-Means Clustering Results with K=3')
plt.scatter(centroids.cluster_centers_[:,0], centroids.cluster_centers_[:,1], marker='+', s=100, c='k', linewidth=2);


b = plt.imread('bird_small.png')

b_re = b.reshape((128*128,3))

b_c = KMeans(16).fit(b_re)

B = b_c.cluster_centers_[b_c.labels_].reshape(128, 128, 3)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13,9))
ax1.imshow(b)
ax1.set_title('Original')
ax2.imshow(B)
ax2.set_title('Compressed, with 16 colors')

for ax in fig.axes:
    ax.axis('off')

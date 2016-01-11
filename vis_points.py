# -*- coding: utf-8 -*-
from __future__ import print_function    # (at top of module)
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn import decomposition, manifold

root_path = '../'
train_data = pd.read_csv(root_path + 'final_train.csv')

def extract_features(data, limit=np.inf):
    X = np.zeros((min(limit, data.shape[0]), 768))#alloc space for the features
    for index, row in data.iterrows():#iterate over csv file
        if index==limit:
            break
        img=cv2.imread(root_path + '/images/' + str(row['image_id']) + '.jpg')
        histogram=np.zeros((3, 256))
        for i in range(3):#calc hist for each channel
            histogram[i] = cv2.calcHist([img],[i],None,[256],[0,256]).ravel()
            cv2.normalize(histogram[i], histogram[i], 0, 1, cv2.NORM_MINMAX)
        X[index]=histogram.ravel()#to 1d array
    return X

X = extract_features(train_data, 500)
y = train_data['image_label'].values[:X.shape[0]].ravel()

pca = decomposition.PCA(n_components=30)
X = pca.fit_transform(X)
tsne = manifold.TSNE(n_components=2, random_state=0)
X = tsne.fit_transform(X)

fig, ax = plt.subplots()
ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='photo')
ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='drawing')

x1=X[y == 1][:, 0]
x2=X[y == 1][:, 1]
xy = np.vstack([x1,x2])
kde = gaussian_kde(xy)#simple density estimation
z = kde(xy)

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

xedges = np.linspace(xmin, xmax, 500)
yedges = np.linspace(ymin, ymax, 500)
xx, yy = np.meshgrid(xedges, yedges)
gridpoints = np.array([xx.ravel(), yy.ravel()])

zz = np.reshape(kde(gridpoints), xx.shape)
im = ax.imshow(zz, cmap='jet', interpolation='nearest', origin='lower', extent=[xmin, xmax, ymin, ymax])
fig.colorbar(im)
plt.legend()
plt.show()

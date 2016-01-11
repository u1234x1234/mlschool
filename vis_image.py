# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold, decomposition
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import gaussian_kde

root_path = '../'
train_data = pd.read_csv(root_path + 'final_train.csv')

def imscatter(x, y, image, ax=None, label=False):
    label=label==True
    im = OffsetImage(image)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=label)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def extract_features(data, limit=np.inf):
    X = np.zeros((min(limit, data.shape[0]), 768))#alloc space for the features
    for index, row in data.iterrows():#iterate over csv file
        if index==limit:
            break
        img=cv2.imread(root_path + '/images/' + str(row['image_id']) + '.jpg')
        histogram=np.zeros((3, 256))
        for i in range(3):#calc hist for each channel
            histogram[i] = cv2.calcHist([img],[i],None,[256],[0,255]).ravel()
            cv2.normalize(histogram[i], histogram[i], 0, 1, cv2.NORM_MINMAX)
        X[index]=histogram.ravel()#to 1d array
    return X

n = 7000
X = extract_features(train_data, n)
y = train_data['image_label'].values[:X.shape[0]].ravel()

X = decomposition.PCA(n_components=50).fit_transform(X)
tsne = manifold.TSNE()
X = tsne.fit_transform(X)

fig, ax = plt.subplots()
scale_factor=15
fig.set_size_inches(16*scale_factor, 9*scale_factor, forward=True)

for index, row in train_data.iterrows():
    if index==n:
        break
    image_path = root_path + '/images/' + str(row['image_id']) + '.jpg'
    image=cv2.imread(image_path)
    b,g,r = cv2.split(image)       # get b,g,r
    image = cv2.merge([r,g,b])     # switch it to rgb
    mx=min(80/image.shape[0],80/image.shape[1])    
    image=cv2.resize(image, (0, 0), image, mx, mx)
    x1=X[index, 0]
    x2=X[index, 1]
    imscatter(x1, x2, image, ax)
    ax.plot(x1, x2)

x1=X[y == 1][:, 0]
x2=X[y == 1][:, 1]
xy = np.vstack([x1,x2])
kde = gaussian_kde(xy)#simple density estimation
z = kde(xy)

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

xedges = np.linspace(xmin, xmax, 700)
yedges = np.linspace(ymin, ymax, 700)
xx, yy = np.meshgrid(xedges, yedges)
gridpoints = np.array([xx.ravel(), yy.ravel()])

zz = np.reshape(kde(gridpoints), xx.shape)
im = ax.imshow(zz, cmap='jet', interpolation='nearest', origin='lower', extent=[xmin, xmax, ymin, ymax])

fig.colorbar(im)
ax.grid()
fig.savefig('7000.png', dpi=100)
#plt.show()



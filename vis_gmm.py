# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn import cluster, manifold, decomposition, metrics, cross_validation, neighbors, pipeline, preprocessing, svm, ensemble, linear_model
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.mixture import GMM

root_path='../'
train_data=pd.read_csv(root_path + 'final_train.csv')
test_data=pd.read_csv(root_path + '/final_test.csv')

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
    X = np.zeros((min(limit, data.shape[0]), 1000))#alloc space for the features
    for index, row in data.iterrows():#iterate over csv file
        if index==limit:
            break
        img=cv2.imread(root_path + '/images/' + str(row['image_id']) + '.jpg')
        img1=cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        histogram = cv2.calcHist([img1],[0,1,2],None,[10,10,10],[0,256,0,256,0,256]).ravel()
        X[index]=histogram.ravel()#to 1d array
    return X

n = 500
X_orig = extract_features(train_data, n)
y = train_data['image_label'].values[:X_orig.shape[0]].ravel()

X = decomposition.PCA(n_components=60).fit_transform(X_orig)
X = manifold.TSNE(random_state=0).fit_transform(X)

def make_ellipses(gmm, ax):
    colors=['r', 'g']
    for i in range(2):
        v, w = np.linalg.eigh(gmm._get_covars()[i][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 0.5
        ell = matplotlib.patches.Ellipse(gmm.means_[i, :2], v[0], v[1], 180 + angle, color=colors[i])
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

gmm = GMM(n_components=2)
gmm.fit(X)
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], cmap='rainbow')
make_ellipses(gmm, ax)
plt.show()


for num_clusters in range(2, 6):
    fig, ax = plt.subplots()

    for index, row in train_data.iterrows():
        if index==n:
            break
        image_path = root_path + '/images/' + str(row['image_id']) + '.jpg'
        image=cv2.imread(image_path)
        b,g,r = cv2.split(image)       # get b,g,r
        image = cv2.merge([r,g,b])     # switch it to rgb
        mx=min(30/image.shape[0],30/image.shape[1])    
        image=cv2.resize(image, (0, 0), image, mx, mx)
        x1=X[index, 0]
        x2=X[index, 1]
        imscatter(x1, x2, image, ax)
        ax.plot(x1, x2)
    
    cl = cluster.KMeans(n_clusters=num_clusters)
    clusters = cl.fit_predict(X)
    
    ax.scatter(X[:, 0], X[:, 1], c=clusters, cmap='rainbow', s=3000)
    ax.scatter(cl.cluster_centers_[:, 0], cl.cluster_centers_[:, 1], s=200, c=range(num_clusters), cmap='rainbow')
    plt.show()
    
cl = cluster.DBSCAN(eps=4, min_samples=10)
clusters = cl.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.show()





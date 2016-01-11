# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd
from sklearn import metrics, cross_validation, neighbors, pipeline, preprocessing, svm, ensemble, linear_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import keras
import numpy as np
import matplotlib.pyplot as plt

root_path='../'
train_data=pd.read_csv(root_path + 'final_train.csv')
test_data=pd.read_csv(root_path + '/final_test.csv')

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

X = extract_features(train_data, 2000)
y = train_data['image_label'].values[:X.shape[0]].ravel()

model_tmp = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1).fit(X, y)
indices = np.argsort(model_tmp.feature_importances_)[::-1]
indices = indices[:200]
X = X[:, indices]

X = np.log2(1 + X)
pre = preprocessing.StandardScaler()
X = pre.fit_transform(X)

num_iter = 100
plt.axis([0, num_iter, 0.5, 1])
plt.grid()

kf=cross_validation.StratifiedKFold(y, n_folds=5)
scores = np.array([])
batch_size = 32
nb_epoch = 30

for train_index, val_index in kf:
    model = Sequential()
    model.add(Dense(128, input_shape=(X.shape[1],), init='uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dense(80, init='uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    opt = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=opt, class_mode='binary')

    xx = np.array([])
    yy_train = np.array([]) 
    yy_val = np.array([])
    color = np.random.rand(3,1)
    for i in range(num_iter):
        model.fit(X[train_index], y[train_index], batch_size=batch_size, 
                  nb_epoch=1, show_accuracy=False, 
                  verbose=0, class_weight={0:9, 1:1})
        preds = model.predict_proba(X[val_index])
        
        auc_roc_val = metrics.roc_auc_score(y[val_index], preds)
        auc_roc_train = metrics.roc_auc_score(y[train_index], model.predict_proba(X[train_index]))

        print ('AUC ROC: ', auc_roc_val)
        yy_val = np.hstack([yy_val, auc_roc_val])
        yy_train = np.hstack([yy_train, auc_roc_train])
        xx = np.hstack([xx, i])
        plt.plot(xx, yy_val, c=color)
        plt.plot(xx, yy_train, '--', c=color)
        plt.pause(0.05)

        
    scores = np.hstack([scores, metrics.roc_auc_score(y[val_index], preds)])
    print ("----------------------------------------------------------------------------")
print ('mean: ', scores.mean(), 'std: ', scores.std())

X_test = pre.transform(np.log2(1 + extract_features(test_data)[:, indices]))
preds = np.zeros((3, X_test.shape[0]))
for i in range(3):
    model = Sequential()
    model.add(Dense(128, input_shape=(X.shape[1],), init='uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dense(80, init='uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    opt = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=opt, class_mode='binary')
    
    model.fit(X, y, batch_size=batch_size, 
                      nb_epoch=50, show_accuracy=False, 
                      verbose=0, class_weight={0:9, 1:1})    
    preds[i] = model.predict_proba(X_test).ravel()

preds = preds.mean(axis=0)

# submission generating
test_data = test_data.drop('image_url', 1)
test_data['image_label'] = preds
test_data.to_csv(root_path + '/res.csv', index=False)# -*- coding: utf-8 -*-


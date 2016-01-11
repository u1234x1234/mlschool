# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd
from sklearn import metrics, cross_validation, neighbors, pipeline, preprocessing, svm, ensemble, linear_model
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
#        трехмерная гистограмма
        histogram = cv2.calcHist([img],[0,1,2],None,[10,10,10],[0,256,0,256,0,256]).ravel()
        X[index]=histogram.ravel()#to 1d array
    return X
    
X = extract_features(train_data, 1000)
y = train_data['image_label'].values[:X.shape[0]].ravel()

#обучение леса
model = ensemble.RandomForestClassifier(n_estimators=100, random_state=0).fit(X, y)

#сортировка по информативности, выбор 40 самых информативных
indices = np.argsort(model.feature_importances_)[::-1]
indices = indices[:40]

#получение компонент цвета ргб по индексу бина в дескрипторе
r = (indices % 10) * 25
g = ((indices // 10) % 10) *25
b = ((indices // 100) % 10) *25
my_colors = [(r[i]/255,g[i]/255,b[i]/255) for i in range(indices.shape[0])]

#визуализация
pos = indices.shape[0] - np.arange(indices.shape[0]) + .5
plt.barh(pos, model.feature_importances_[indices], align='center', color=my_colors)
plt.yticks(pos, indices)
plt.title('Variable Importance')
plt.show()

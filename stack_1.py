# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics, cross_validation, neighbors, pipeline, preprocessing, svm, ensemble, linear_model
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
import xgboost as xgb

root_path='./'
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

X = extract_features(train_data)
y = train_data['image_label'].values[:X.shape[0]].ravel()

#model_tmp = ensemble.RandomForestClassifier(n_estimators=500, n_jobs=-1).fit(X, y)
#indices = np.argsort(model_tmp.feature_importances_)[::-1]
#indices = indices[:400]
#X = X[:, indices]

meta_model = linear_model.LogisticRegression()

class StackClassifier():
    def __init__(self):
        self.models = [
        ensemble.RandomForestClassifier(n_estimators=1000, n_jobs=-1),
        Pipeline([('pre', Normalizer()), ('svm', svm.SVC(kernel='linear', C=10, probability=True))]),
        Pipeline([('pre', Normalizer(norm='l1')), ('svm', KNeighborsClassifier(n_neighbors=40, p=1, n_jobs=-1))]),
        ensemble.AdaBoostClassifier(),
        linear_model.LogisticRegression(),
        xgb.sklearn.XGBClassifier(max_depth=10, n_estimators=2000, colsample_bytree=0.9, subsample=0.9, learning_rate=0.03)
        ]
    def fit(self, X, y):
        for i in range(len(self.models)):
            self.models[i].fit(X, y)
    def get_features(self, X):
        models_preds = np.array([])
        for i in range(len(self.models)):
            pred = self.models[i].predict_proba(X)[:, 1]
            models_preds = np.vstack([models_preds, pred]) if models_preds.size else pred
        return models_preds.T

kf=cross_validation.StratifiedKFold(y, n_folds=5)
scores = np.array([])
for train_index, val_index in kf:
    X_1, X_2, y_1, y_2 = train_test_split(X[train_index], y[train_index], train_size=0.7)
    
    st = StackClassifier()
    st.fit(X_1, y_1)
    level_2_features = st.get_features(X_2)
    scores= cross_validation.cross_val_score(meta_model, level_2_features, y_2, scoring='roc_auc', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    meta_model.fit(level_2_features, y_2)
    print (meta_model.coef_, meta_model.intercept_)
    preds = meta_model.predict_proba(st.get_features(X[val_index]))[:, 1]
    print ('AUC ROC: ', metrics.roc_auc_score(y[val_index], preds))
    scores = np.hstack([scores, metrics.roc_auc_score(y[val_index], preds)])
print ('mean: ', scores.mean(), 'std: ', scores.std())

st = StackClassifier()
st.fit(X[:11000], y[:11000])
meta_model.fit(st.get_features(X[11000:]), y[11000:])
preds = meta_model.predict_proba(st.get_features(extract_features(test_data)[:, :]))[:, 1]

test_data = test_data.drop('image_url', 1)
test_data['image_label'] = preds
test_data.to_csv(root_path + '/res.csv', index=False)# -*- coding: utf-8 -*-


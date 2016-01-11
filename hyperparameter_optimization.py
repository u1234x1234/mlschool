from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.pipeline import Pipeline
import cv2
import numpy as np
import pandas as pd
from sklearn import neighbors, preprocessing

root_path='../'
train_data=pd.read_csv(root_path + 'final_train.csv')
test_data=pd.read_csv(root_path + '/final_test.csv')

def extract_features(data, limit=np.inf):
    X = np.zeros((min(limit, data.shape[0]), 768))#alloc space for the features
    for index, row in data.iterrows():#iterate over csv file
        if index==limit:
            break
        img=cv2.imread(root_path + '/images/' + str(row['image_id']) + '.jpg')
        histogram=np.zeros((3, 256))
        for i in range(3):#calc hist for each channel
            histogram[i] = cv2.calcHist([img],[i],None,[256],[0,255]).ravel()
        X[index]=histogram.ravel()#to 1d array
    return X
    
X = extract_features(train_data, 1500)
y = train_data['image_label'].values[:X.shape[0]].ravel()
    
grid = {
    'knn__n_neighbors': [1, 10, 20, 30, 40, 60, 75, 100, 120, 160, 200],
    'knn__metric': ['euclidean', 'manhattan', 'chebyshev'],
    'knn__weights': ['uniform', 'distance'],
    'preprocess__norm': ['l1', 'l2', 'max']
}
pipeline = Pipeline(steps=[
    ('preprocess', preprocessing.Normalizer()),
    ('knn', neighbors.KNeighborsClassifier())
])

model = EvolutionaryAlgorithmSearchCV(pipeline, grid, scoring='roc_auc', verbose=True, n_jobs=4, population_size=10)
model.fit(X, y)

preds = model.predict_proba(extract_features(test_data))[:, 1]
test_data = test_data.drop('image_url', 1)
test_data['image_label'] = preds
test_data.to_csv(root_path + '/res.csv', index=False)
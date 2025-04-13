import numpy as np
import pandas as pd
from collections import defaultdict

def euclideanDistance(vecA: np.array, vecB:np.array):
    return np.sqrt(sum((vecA-vecB)**2))

def manhattanDistance(vecA: np.array, vecB: np.array):
    return np.sum(abs(vecA-vecB))

class KNNCLassifier:
    def __init__(self, k=5, distance='euclidean'):
        self.k = k
        self.dist = distance

    def fit(self,X: np.array,y: pd.Series):
        self.Xtrain = X
        if self.Xtrain.shape[0] < self.k:
            self.k = self.Xtrain.shape[0]
        self.ytrain = y
        self.labelOrder = np.unique(y)
        self.mapLabel = {i:l for i,l in enumerate(self.labelOrder)}
        self.mapLabel = defaultdict(lambda: 'Not Found Label',self.mapLabel)

    def predict_proba(self,X: np.array):
        if X.shape[1] != self.Xtrain.shape[1]:
            raise IndexError('Expected size wasnt solved, check size')
        distFun = euclideanDistance if self.dist == 'euclidean' else manhattanDistance
        probs = []
        for r in X:
            distValues = [distFun(r,t) for t in self.Xtrain]
            idxs = np.argsort(distValues) #Ascending sort
            bestK = self.ytrain[idxs[:self.k]]
            probs.append([sum(bestK==l)/self.k for l in self.labelOrder])
        return np.array(probs)
    
    def predict(self,X: np.array):
        predictProbas = self.predict_proba(X)
        return np.array([self.mapLabel[np.argmax(k)] for k in predictProbas])

class KNNRegressor:
    def __init__(self, k=5, distance='euclidean'):
        self.k = k
        self.dist = distance

    def fit(self, X: np.array, y: pd.Series):
        self.Xtrain = X
        if self.Xtrain.shape[0] < self.k:
            self.k = self.Xtrain.shape[0]
        self.ytrain = y

    def predict(self, X: np.array):
        if X.shape[1] != self.Xtrain.shape[1]:
            raise IndexError('Expected size wasnt solved, check size')
        distFun = euclideanDistance if self.dist == 'euclidean' else manhattanDistance
        values = []
        for r in X:
            distValues = [distFun(r, t) for t in self.Xtrain]
            idxs = np.argsort(distValues)  # Ascending sort
            bestK = self.ytrain[idxs[:self.k]]
            values.append(np.mean(bestK))
        return np.array(values)

data = pd.read_csv(r'databases\Iris.csv')
X = data.drop(['species'], axis=1).values
y = data['species'].values
knClass  = KNNCLassifier(k=1,distance='euclidean')
knClass.fit(X,y)
print(sum(knClass.predict(X)==y))
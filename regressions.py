import numpy as np
import pandas as pd
from math import ceil

class LinearRegression:
    def __init__(self):
        self.fitted = False

    def fit(self,X: np.array, y: pd.Series):
        X = np.c_[np.ones(shape=X.shape[0]),X]
        left = np.linalg.inv(X.T.dot(X))
        right = X.T.dot(y)
        self.coef_ = left.dot(right)
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:].T
        self.fitted = True

    def correctShape(self,X: np.array):
        return X.shape[1]==len(self.coef_)

    def predict(self,X: np.array):
        if not self.correctShape(X):
            raise ValueError(f'Shape doesn\'t match, correct shape is {len(self.coef_)}')
        return self.intercept_ + X.dot(self.coef_)

class SGDRegressor:
    def __init__(self, penalty: str = None, batch_size: int = 5, max_iter: int =1000, 
                min_tol: float =1e-3, alpha: float=0.01, lamda: float = 0.1):
        self.penalty = penalty
        self.batch_size = batch_size
        self.mxIter = max_iter
        self.tol = min_tol
        self.lr = alpha
        self.lambd = lamda

    def correctShape(self,X: np.array):
        return X.shape[1]==len(self.coef_)

    def updateCoef(self,errors: np.array):
        penalt = 0
        if self.penalty=='l1':
            penalt = sum(abs(self.coef_))
        elif self.penalty=='l2':
            penalt = sum(self.coef_**2)

        self.coef_ = -1*(self.lr*errors+penalt*self.lambd).reshape(-1,1)+self.coef_
        
    def fit(self,X: np.array, y:pd.Series, verbose=0):
        n = X.shape[0]
        X = np.c_[np.ones(n),X]
        self.coef_ = np.ones(shape=(X.shape[1],1))
        outerBreak = False
        for epoq in range(self.mxIter):
            for bt in range(ceil(self.batch_size/n)):
                currErr = np.array([])
                lowerLim = bt*self.batch_size
                upperLim = min(n,lowerLim+self.batch_size)
                currErr = -2*(X[lowerLim:upperLim].T.dot(y[lowerLim:upperLim].T-X[lowerLim:upperLim].dot(self.coef_))) # Calcular errores
                currErr = np.mean(currErr,axis=1)
                self.updateCoef(currErr)
                if max(abs(currErr))<self.tol:
                    outerBreak = True
                    break
            if outerBreak:break
        if verbose:
            print('Number of Iterations:',(epoq+1))
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]
    
    def predict(self,X: np.array):
        if not self.correctShape(X):
            raise ValueError(f'Shape doesn\'t match, correct shape is {len(self.coef_)}')
        return self.intercept_ + X.dot(self.coef_)

def sigmoid(x):
    return 1/(1+np.exp(x))

class LogisticRegression:
    def __init__(self, penalty: str = None, batch_size: int = 5, max_iter: int =1000,
                min_tol: float =1e-3, alpha: float=0.01, lamda: float = 0.1):
        self.penalty = penalty
        self.batch_size = batch_size
        self.mxIter = max_iter
        self.tol = min_tol
        self.lr = alpha
        self.lambd = lamda
    
    def correctShape(self,X: np.array):
        return X.shape[1]==len(self.coef_)

    def updateCoef(self,errors: np.array):
        penalt = 0
        if self.penalty=='l1':
            penalt = sum(abs(self.coef_))
        elif self.penalty=='l2':
            penalt = sum(self.coef_**2)

        self.coef_ = -1*(self.lr*errors+penalt*self.lambd).reshape(-1,1)+self.coef_

    def transformY(self, y:pd.Series):
        y = np.array(y).reshape(-1,1)
        if len(np.unique(y))!=2:
            raise ValueError('There must be only two labels')
        if 1 in y and 0 in y:
            self.labels = {1:1,0:0}
            return y
        y[y==np.unique(y)[0]] = 1
        y[y==np.unique(y)[1]] = 0
        self.labels = {i:val for i,val in enumerate(np.unique(y))}
        return y

    def fit(self,X: np.array, y:pd.Series, verbose=0):
        n = X.shape[0]
        X = np.c_[np.ones(n),X]
        y = self.transformY(y)
        self.coef_ = np.ones(shape=(X.shape[1],1))
        outerBreak = False
        for epoq in range(self.mxIter):
            for bt in range(ceil(self.batch_size/n)):
                currErr = np.array([])
                lowerLim = bt*self.batch_size
                upperLim = min(n,lowerLim+self.batch_size)
                currErr = -2*(X[lowerLim:upperLim].T.dot(y[lowerLim:upperLim].T-sigmoid(X[lowerLim:upperLim]).dot(self.coef_)))
                if np.any(np.isnan(currErr)):
                    currErr[np.isnan(currErr)] = 0 
                currErr = np.mean(currErr,axis=1)
                self.updateCoef(currErr)
                if max(abs(currErr))<self.tol:
                    outerBreak = True
                    break
            if outerBreak:break
        if verbose:
            print('Number of Iterations:',(epoq+1))
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def predict_proba(self,X: np.array):
        if not self.correctShape(X):
            raise ValueError(f'Shape doesn\'t match, correct shape is {len(self.coef_)}')
        return sigmoid(self.intercept_ + X.dot(self.coef_))

    def predict(self,X: np.array, labels=True):
        predicted = np.round(self.predict_proba(X))
        if labels:
            predicted = np.array(map(lambda m:self.labels[m],predicted))
        return predicted

#### Ejemplo Regresión Logística

# data = pd.read_csv('train_and_test2.csv')
# data = data[[col for col in data.columns if not 'zero' in col]]
# X = data.drop(columns=['2urvived','Passengerid'])
# y = data['2urvived']

# logreg = LogisticRegression()
# logreg.fit(X,y, verbose=1)

# print(np.c_[X.columns,logreg.coef_])

#### Ejemplo Regresión Lineal

data = pd.read_csv('Linear Regression - Sheet1.csv')
X = data.drop(columns='Y')
Y = data['Y']

linreg = SGDRegressor() # LinearRegression
linreg.fit(X,Y)

print(linreg.coef_,linreg.intercept_)

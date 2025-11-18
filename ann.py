# Creation of Artificial Neural Networks with manual code
# Author: Carlos Guayambuco
print('Staert')
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from random import random
from tqdm import tqdm
from math import ceil
import matplotlib.pyplot as plt

class NNFunc:
    def __init__(self, func, derivate):
        self.func = func
        self.der = derivate

    def __call__(self, x, **kwds):
        if len(x.shape) != 1:
            return np.array([self(i) for i in x])
        return np.array(self.func(x))
    
    def derivate(self, x):
        if len(x.shape) != 1:
            return np.array([self.derivate(i) for i in x])
        return np.array(self.der(x)) 

    def __repr__(self):
        return 'name'

class Relu(NNFunc):
    def __init__(self):
        def func(x): return np.array([0 if i < 0 else i for i in x]) 
        def derivate(x): return np.array([0 if i < 0 else 1 for i in x])
        super().__init__(func, derivate)

    def __repr__(self):
        return 'Relu'

class Sigmoid(NNFunc): # Asyntote problems
    def __init__(self):
        self.func = lambda x: 1/(1+np.exp(-x))
        def derivate(x): return self.func(x)*(1-self.func(x))
        super().__init__(self.func, derivate)
    
    def __repr__(self):
        return 'Sigmoid'

class ANN:
    def __init__(self, numNeur: list[int], funAct: list[str], lr: float = 0.01):
        """Initializes the network that will be used

        Args:
            numNeur (list[int]): A list containing number of neurons per layer
            funAct (list[str]): The activation function per layer, with input as a string.
            lr (int): The learning rate that will be used for updating the parameters
        
        **Disclaimer**: Input neurons are required
        """
        assert len(numNeur)-1 == len(funAct),f"Number of layers doesn't match, {len(numNeur)-1} != {len(funAct)}"
        self.nNeur = numNeur
        self.fAct = funAct
        self.lr = lr
        self.isTrained = False
        self._initLayers()
        self._initFuncs()

    def __call__(self, *args, **kwds):
        return self._forward(args[0])

    def _initLayers(self, mode: str = 'rand'):
        """Initializes the layers of the network

        Args:
            mode (str, optional): Eather if the weights are decided randomly or 1s. Defaults to 'rand'.
        """        
        self.layers = []
        self.biases = []
        for i in range(len(self.nNeur)-1):
            inpDim = self.nNeur[i]
            outDim = self.nNeur[i+1]
            if mode == 'rand':
                self.layers.append(np.random.random(size=(inpDim, outDim)))
                self.biases.append(np.random.random(size=(outDim,1)))
            else:
                self.layers.append(np.ones((inpDim, outDim)))
                self.biases.append(np.ones((outDim, 1)))

    def _initFuncs(self):
        """Initializes the current functions that will be used
        """        
        self.funcs = {'relu':Relu(),'sigmoid':Sigmoid()}

    def _initLoss(self):
        """Initializes both loss and it's derivate that will be used by the model
        """        
        if self.loss == 'MSE':
            self.lossF = lambda y_T, y_P: 1/2*((y_T-y_P)**2)
            self.derF = lambda y_T, y_P: y_P - y_T
        elif self.loss == 'CE':
            self.auxF = lambda x: round(abs(x), 4) == 0 or round(abs(x), 4) == 1
            self.lossF = lambda y_T, y_P: -1*(y_T*np.log(y_P))-(1-y_T)*np.log(1-y_P) if self.auxF(y_P) else 100
            self.derF = lambda y_T, y_P: -1*(y_T/y_P) - ((1-y_T)/(1-y_P)) if self.auxF(y_P) else 100
        self.lossF = np.vectorize(self.lossF) # Convert functions into mapping funcs
        self.derF = np.vectorize(self.derF)

    def _transformY(self, y) -> np.array:
        """Given a y vector transforms it into a one hot vector

        Args:
            y (array like): The possible Values

        Returns:
            np.array: An array with one hot version of y 
        """        
        uniqVals = {v:i for i, v in enumerate(np.unique(y))}
        greatMat = np.zeros((len(y),len(uniqVals))) # A zeroes matrix for encoding
        for i in range(len(y)):
            idx = uniqVals[y[i]]
            greatMat[i][idx] = 1
        return greatMat

    def _forward(self, x) -> np.array: # Completed
        """The forward pass of the model

        Args:
            x (Array Like): The data for the model

        Returns:
            Array: The final result
        """        
        self.acts = [x] # The activations of each layer
        self.zs = []
        for l in range(len(self.layers)):
            newb = np.hstack([self.biases[l] for _ in range(len(x))]).T
            z = x @ self.layers[l] + newb
            usedFunc = self.funcs[self.fAct[l].lower()]
            a = usedFunc(z)
            x = a
            self.zs.append(z)
            self.acts.append(a)
        return x
    
    def _backward(self, y): # Completed
        """Uses the backpropagation algorithm to find the optimal weights and biases

        Args:
            y (Array Like): The true values of the data given
        """        
        derA = self.derF(y, self.acts[-1]).reshape((-1, 1))
        for l in range(len(self.layers))[::-1]:
            # Calculating Derivates
            usedFunc = self.funcs[self.fAct[l].lower()]
            derAZ = np.multiply(derA, usedFunc.derivate(self.zs[l])) / self.btSize # Size n x p_a Good
            derW = self.acts[l].T @ derAZ  # Size p_(a-1) x p_a Good
            derB = np.mean(derAZ, axis=0).reshape((-1,1)) # Good
            derA = derAZ @ self.layers[l].T # Good

            # Updating Weights
            self.layers[l] -= self.lr*derW
            self.biases[l] -= self.lr*derB

    def lossVal(self, X, y) ->  float:
        """The current loss

        Args:
            X (Array Like): The data for prediction
            y (Array Like): The real data

        Returns:
            float: The total loss of the network
        """        
        pred = self._forward(X)
        return np.sum(self.lossF(y, pred))

    
    def train(self, X, y, epochs: int = 100, batch_size: int = 5, lossFunc: str = 'MSE',normY = True):
        """Trains the neural network

        Args:
            X (Array Like): The data for training
            y (Array Like): The true values of the data
            epochs (int, optional): Number of epochs for training. Defaults to 100.
            batch_size (int, optional): The batch size. Defaults to 5.
            lossFunc (str, optional): The loss function that will be used. Defaults to 'MSE'.
            normY (bool, optional): Normalize y as an one hot vector. Defaults to True.
        """        
        assert X.shape[0] == len(y), f"Rows of X and y don't match: {X.shape[0]} != {len(y)}"
        self.loss = lossFunc
        self.normY = normY
        self._initLoss()
        self.isTrained = True
        if lossFunc == 'CE' and normY:
            y = self._transformY(y)
        else:
            y = y.reshape((-1,1))
        for ep in tqdm(range(epochs)): 
            # Each Epoch
            for b in range(ceil(X.shape[0]/batch_size)): 
                # Each Batch
                batchX = X[b*batch_size:min((b+1)*batch_size,X.shape[0])]
                batchy = y[b*batch_size:min((b+1)*batch_size, X.shape[0])]
                self.btSize = min(batch_size, len(batchy))
                # Forward Pass
                self._forward(batchX)
                # Backward Pass
                self._backward(batchy)

        print('Training Complete')
        print('Total Loss:',self.lossVal(X,y))

    def pred_proba(self, X) -> np.array:
        """Returns the predicted probability for each data

        Args:
            X (Array Like): The data for the predictions

        Returns:
            np.array: An array with the probabilities or final values
        """        
        return self._forward(X)

    def predict(self, X) -> np.array:
        """Predict the final values with the current network

        Args:
            X (Array Like): The data for the prediction

        Returns:
            np.array: An array with the final predicted values
        """        
        if self.loss == 'CE':
            if self.normY:
                return np.max(self._forward(X),axis=0)
            return np.round(self._forward(X))
        return np.round(np.abs(self._forward(X)))

    def showConvergence(self,X,y, dim: str = '2d', numPoints: int = 100,vmin = -5, vmax=5, seed: int = None) -> None:
        """This is my addon of the model library, a way to visualize the convergence of the model,
        which is seen by adding a random direction alpha along different factor values. Looking for the 
        valley of the random directions we could judge if it's optimal or not

        Args:
            X (_type_): The values for predicting with the ANN
            y (_type_): The real values
            dim (str, optional): The dimension of the graph, 2d or 3d. Defaults to '2d'.
            vmin (optional): The minimum value for the factor of alpha
            vmax (optional): The maximum value for the factor of alpha
            numPoints (int, optional): Number of points utilized among each axis. Defaults to 100.
        """        
        import plotly.express as px
        import plotly.graph_objects as go


        # Both Dimensions Working
        if not seed: # Setting the seed
            seed = np.random.rand()
        y = y.reshape((-1, 1))

        # Having original parameters
        originalLayers = [arr.copy() for arr in self.layers]
        originalBiases = [arr.copy() for arr in self.biases]

        # Creating the alphas for weights
        a1, a2 = [],[]
        # Creating alphas for intercept
        ab1, ab2 = [],[]
        for i in range(len(self.layers)):
            a1.append(np.random.random(size = self.layers[i].shape))
            ab1.append(np.random.random(size = self.biases[i].shape))
            if dim == '3d':
                a2.append(np.random.random(size = self.layers[i].shape))
                ab2.append(np.random.random(size = self.biases[i].shape))

        posibleFactors = np.linspace(vmin,vmax,numPoints)
        posibleFactors = np.round(posibleFactors,5)
        data = {'f1':[],'loss':[]}
        if dim == '3d':
            data['f2'] = []

        # Evaluating each combination
        if dim == '2d':
            for p1 in posibleFactors: # 2d Working as Intended
                newLayers = [arr.copy() for arr in originalLayers]
                newBiases = [arr.copy() for arr in originalBiases]
                for j in range(len(originalLayers)):
                    newLayers[j] += p1*a1[j]
                    newBiases[j] += p1*ab1[j]
                data['f1'].append(p1)
                self.layers = newLayers
                self.biases = newBiases
                data['loss'].append(self.lossVal(X, y))
        if dim=='3d':
            for p1 in posibleFactors:
                for p2 in posibleFactors:
                    newLayers = [arr.copy() for arr in originalLayers]
                    newBiases = [arr.copy() for arr in originalBiases]
                    for k in range(len(originalLayers)):
                        newLayers[k] += p2*a2[k] + p1*a1[k]
                        newBiases[k] += p2*ab2[k] + p1*ab1[k]
                    data['f1'].append(p1)
                    data['f2'].append(p2)
                    self.layers = newLayers
                    self.biases = newBiases
                    data['loss'].append(self.lossVal(X,y))

        data = pd.DataFrame(data)
        print('Min Loss:',data.sort_values(by='loss',ascending=True).iloc[0])

        if dim == '2d': # 2d Plot
            fig = px.line(data,x='f1',y='loss',title="Valley of Loss")
        elif dim == '3d':
            pivotDf = data.pivot_table(index='f2',columns='f1',values='loss')
            figdata = go.Surface(x=pivotDf.columns.values,
                                y=pivotDf.index.values,
                                z=pivotDf.values,colorscale='inferno')
            fig = go.Figure(data=[figdata])
            fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                highlightcolor="limegreen", project_z=True))
            fig.update_layout(title='3D Convergence Valley',
                    scene=dict(xaxis_title='a1',
                                yaxis_title='a2',
                                zaxis_title='loss'))
        fig.show()
        self.layers = originalLayers # Return layers to trained state
        self.biases = originalBiases # Return biases to trained state

firstAnn = ANN([4,2,1],['relu','sigmoid'],lr=0.05)


data = pd.read_csv('databases/iris.csv')
X = data.drop('species',axis=1).values
y = np.array(data['species'] == 'setosa').astype(int).reshape((-1, 1))

firstAnn.train(X,y,lossFunc='MSE',epochs=1000, normY=False)
print('Ready')
firstAnn.showConvergence(X,y,numPoints=111,vmin=-4,vmax=4,dim='3d')
print('Accuracy:',accuracy_score(y,np.round(firstAnn(X))))
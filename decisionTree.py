import numpy as np
from collections import defaultdict, Counter
import pandas as pd
import time
import matplotlib.pyplot as plt
from statistics import mode
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from matplotlib import patches

def calcEntropy(cntr: dict[int]) -> float:
    """Calculates the Entropy for a given Counter

    Args:
        cntr (dict[int]): A counter-like Object for calculating the entropy

    Returns:
        float: Return the value of the Entropy
    """
    allVals = sum(cntr.values())
    if not allVals:
        return 0
    return -1*sum(np.log2(c/allVals)*(c/allVals) for c in cntr.values() if c)

def calcGini(cntr: dict[int]) -> float:
    """Calculates the Gini Index given the Counter
    Args:
        cntr (dict[int]): A counter-like Object for calculating the gini index

    Returns:
        float: The Gini Index Value
    """
    allVals = sum(cntr.values())
    if not allVals:
        return 0
    return 1-sum((c/allVals)**2 for c in cntr.values() if c)

def bestNumSplit(numVar: np.array,objVar: np.array, gainInfo=True) -> tuple[float,float]:
    """Finds the best split given the stuff

    Args:
        numVar (np.array[float]): The array with all the numeric info
        objVar (np.array[float]): The array with the objective variable
        gainInfo (bool): If gain Info is used for findinf best split. Defaults to True.

    Returns:
        tuple[float,float]: Returns the best split value as well as the best Metric
    """
    data = pd.DataFrame({'numVar':numVar,'objVar':objVar}).sort_values(by='numVar') # Sort the data
    rightCntr = defaultdict(int) # Create counters for calc Gini or Entropy
    bestMetric = 0
    best_split = numVar[0]
    for c in objVar:
        rightCntr[c] += 1 # Initializing Counter
    totalVal = calcEntropy(rightCntr) if gainInfo else calcGini(rightCntr)
    leftCntr = defaultdict(int)
    leftSize, rightSize = 0,len(numVar)
    for _, row in data.iterrows(): # Itering for every split
        rightCntr[row['objVar']] -= 1 # Changing the counters
        leftCntr[row['objVar']] += 1
        leftSize += 1
        rightSize -= 1
        leftWeight = calcEntropy(leftCntr) if gainInfo else calcGini(leftCntr) # Calculating new Metric
        rightWeight = calcEntropy(rightCntr) if gainInfo else calcGini(rightCntr)
        if gainInfo:
            auxGain = totalVal-(leftWeight*leftSize+rightWeight*rightSize)/len(numVar)
        else:
            auxGain = (leftWeight*leftSize+rightWeight*rightSize)/len(numVar)
        if  (auxGain > bestMetric and gainInfo) or (auxGain < bestMetric and not(gainInfo)):
            # IF metric is best change best metric
            bestMetric = auxGain
            best_split = row['numVar']

    return best_split, bestMetric

def bestCatSplit(catVar: np.array,objVar: np.array,gainInfo: bool = True) -> tuple[bool,float]:
    """Takes in Count if it's a good variable to split as well as returning the gainInfo or the GiniIndex

    Args:
        catVar (np.array[float]): The array with all the categoric info
        objVar (np.array[float]): The array with the objective variable
        gainInfo (bool): If gain Info is used for findinf best split. Defaults to True.

    Returns:
        tuple[bool,float]: Tuple with a Bool representing if it's good for splitting and float as the metric Value
    """
    cntr = Counter(objVar)
    catSets = [objVar[(catVar==c)] for c in np.unique(catVar)] # Get the unique sets for each category
    catSizes = [len(s) for s in catSets] # Get the sizes
    totalVal = calcEntropy(cntr) if gainInfo else calcGini(cntr) # Calc the initial Metric
    newMetric = [calcEntropy(catSets[i]) if gainInfo else calcGini(catSets[i]) for i in range(len(catSets))] # Calculate
    newVal = sum(newMetric[i]*catSizes[i] for i in range(len(catSizes)))/len(objVar) # new submetrics and create the 
    result = [False,0]                                                               # final new Metric
    if newMetric < totalVal: # Comparing if it's a better Metric
        result[1] = totalVal-newVal if gainInfo else newVal
        result[0] = True
    return tuple(result)

class Leaf:
    def __init__(self, value):
        self.strCond = value

    def predict(self,*args):
        return self.strCond
    
    def __str__(self):
        return 'Leaf '+str(self.strCond)

class Node:
    def __init__(self, funcCondition, nxt, strCond = ''):
        self.cond = funcCondition
        self.next = nxt
        self.strCond = strCond

    def predict(self, val):
        if self.cond(val):
            if isinstance(self.next,Leaf):
                return self.next.predict(val)
            predictions = list(filter(None,[n.predict(val) for n in self.next]))
            return predictions[0] if predictions else None
        return None
    
    def __str__(self):
        return 'Node '+self.strCond

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.mxDp = max_depth

    def findBestSplitter(self,X: np.array,y: np.array):
        totalMetrics = []
        for i in range(len(X[0])):
            column = X[:,i].flatten()
            if self.colTypes[i]=='cat':
                totalMetrics.append(bestCatSplit(column,y,self.asm))
            else:
                totalMetrics.append(bestNumSplit(column,y,self.asm))
        return totalMetrics

    def _fit(self,X,y,depth = 0):
        if len(np.unique(y))==1:
            self.depthCount[depth] = self.depthCount.get(depth, 0)+1
            return Leaf(y[0])
        splitters= self.findBestSplitter(X,y)
        splitInfo, splitValues = zip(*splitters)
        bestSplit = np.argmax(splitValues) if self.asm else np.argmin(splitValues)
        colInfo = X[:,bestSplit].flatten()
        ## Case when there worst splits
        if (splitValues[bestSplit]==0.5 and not(self.asm)) or (splitValues[bestSplit]==0 and self.asm) or depth==self.mxDp:
            self.depthCount[depth] = self.depthCount.get(depth,0)+1
            self.depthCount[depth+1] = self.depthCount.get(depth+1, 0)+1
            return [Node(lambda x: True,Leaf(mode(y)),strCond='MaxDepth' if depth==self.mxDp else 'No Split')]

        ## Case for Basic Splits

        if self.colTypes[bestSplit] == 'cat':
            if not splitInfo[bestSplit]:
                self.depthCount[depth] = self.depthCount.get(depth, 0)+1
                self.depthCount[depth+1] = self.depthCount.get(depth+1, 0)+1
                return [Node(lambda x: True,Leaf(mode(y)),strCond='No Split')]
            retList = []
            for c in np.unique(colInfo):
                actualNode = Node(lambda x: x[bestSplit]==c, self._fit(X[(colInfo==c)],y[(colInfo==c)],depth+1),strCond=f'Column {bestSplit} == {c}') 
                retList.append(actualNode)
            self.depthCount[depth] = self.depthCount.get(depth, 0)+len(retList)
            return retList
        else:
            bestVal = splitInfo[bestSplit]
            if sum(colInfo>bestVal):
                actualNode = Node(lambda x: x[bestSplit]<=bestVal, self._fit(X[(colInfo<=bestVal)],y[(colInfo<=bestVal)],depth+1), strCond=f'Column {bestSplit} <= {bestVal}')
                oppositeNode = Node(lambda x: x[bestSplit]>bestVal, self._fit(X[(colInfo>bestVal)],y[(colInfo>bestVal)],depth+1), strCond=f'Column {bestSplit} > {bestVal}')
                retList = [actualNode, oppositeNode]
            else:
                self.depthCount[depth] = self.depthCount.get(depth, 0)+1
                self.depthCount[depth+1] = self.depthCount.get(depth+1, 0)+1
                return [Node(lambda x: True, Leaf(mode(y)), strCond='No Split')]
            self.depthCount[depth] = self.depthCount.get(depth, 0)+2
            return retList


    def createcolTypes(self,X):
        cols = []
        for i in range(len(X[0])):
            column = X[:,i].flatten()
            if np.issubdtype(column.dtype, np.int_) or np.issubdtype(column.dtype, np.str_):
                cols.append('cat')
            else:
                cols.append('num')
        self.colTypes = cols

    def fit(self,X,y, asm: str ='infoGain'):
        # X categorical Features must be passed as objects or ints
        self.createcolTypes(X)
        self.asm = (asm == 'infoGain')
        self.depthCount = {0:1}
        self.model = Node(lambda x: True,self._fit(X,y.flatten(),1),strCond='Root')

    def predict(self,val):
        if len(val.shape)>1:
            return [self.predict(val[i]) for i in range(val.shape[0])]
        return self.model.predict(val)
    
    def drawTree(self, squareSize=100):
        fig, ax = plt.subplots(figsize=(10,10))
        self.sqrSz = squareSize
        maxDepth = max(self.depthCount.keys())+1
        maxAmplitude = max(self.depthCount.values())
        self.depthCopy = self.depthCount.copy()
        self.depthAmpSize = {}
        for k,v in self.depthCount.items():
            self.depthAmpSize[k] = (maxAmplitude*(3*self.sqrSz//2)-self.sqrSz*v)//(v+1)
        ax.set_xlim(0, (maxAmplitude*(3*self.sqrSz//2)))
        ax.set_ylim(self.sqrSz//2,(maxDepth*(3*self.sqrSz//2))+self.sqrSz//2)
        self._drawTree(self.model,ax,0)
        plt.show()

    def _drawTree(self,node,ax,depth,lastCoords=False):

        xPos = (self.depthCount[depth]-self.depthCopy[depth]+1)*(self.depthAmpSize[depth])
        xPos += (self.depthCount[depth]-self.depthCopy[depth])*self.sqrSz
        self.depthCopy[depth]-=1
        yPos = depth*self.sqrSz+self.sqrSz//2*(depth+1)
        coords = (xPos, yPos)
        ax.add_patch(patches.Rectangle(coords,self.sqrSz,self.sqrSz,fill=True,color='blue' if isinstance(node,Node) else 'red'))
        ax.text(xPos+self.sqrSz//2,yPos+self.sqrSz//2,node.strCond,ha='center',va='center')

        if not isinstance(node,Leaf) and  not isinstance(node.next,Leaf):
            for n in node.next:
                newCoords = self._drawTree(n, ax, depth+1, coords)
                ax.plot([xPos+self.sqrSz//2, newCoords[0]+self.sqrSz//2], [yPos+self.sqrSz, newCoords[1]], color='black')
        elif not isinstance(node, Leaf):
            newCoords = self._drawTree(node.next,ax,depth+1,coords)
            ax.plot([xPos+self.sqrSz//2,newCoords[0]+self.sqrSz//2],[yPos+self.sqrSz,newCoords[1]],color='black')

        return xPos,yPos


# Prueba Clasificador Iris solo variables Númericas
myClassifier = DecisionTreeClassifier(max_depth=5)
myData = pd.read_csv('databases\Iris.csv')
X = myData.drop(['species'],axis=1).values
y = myData['species'].values
start = time.time()
myClassifier.fit(X,y)
stats = time.time()-start
print('Tiempo Total para Iris:',stats)
# print(classification_report(y,myClassifier.predict(X)))
myClassifier.drawTree(squareSize=200)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# myClassifier.fit(X_train,y_train)
# print(classification_report(y_test,myClassifier.predict(X_test)))

# # Prueba Clasificador Drogas variables númericas y categóricas
# myClassifier = DecisionTreeClassifier(max_depth=5)
# myData = pd.read_csv('drug200.csv')
# X = myData.drop(['Drug'],axis=1).values
# y = myData['Drug'].values
# start = time.time()
# myClassifier.fit(X,y)
# stats = time.time()-start
# print('Tiempo Total para Drug Test:',stats)
# print(classification_report(y,myClassifier.predict(X)))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# myClassifier.fit(X_train,y_train)
# print(classification_report(y_test,myClassifier.predict(X_test)))

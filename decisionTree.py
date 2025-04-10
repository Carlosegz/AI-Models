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

def mseMod(y):
    return sum((y-np.mean(y))**2)

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
    def __init__(self, max_depth=None, min_point=3):
        self.mxDp = max_depth
        self.minPt = min_point
    
    def bestCatSplit(self ,catVar: np.array,objVar: np.array) -> tuple[str,float]:
        cntr = Counter(objVar)
        totalVal = calcEntropy(cntr) if self.asm else calcGini(cntr) # Calc the initial Metric
        bestMetric = 0 if self.asm else 0.5
        result = [False,bestMetric]   
        if len(cntr) == 1:
            return result
        for c in np.unique(catVar):
            left = catVar==c
            right = catVar!=c
            if sum(left) < self.minPt or sum(right) < self.minPt:
                continue
            leftCntr = Counter(objVar[left])
            rightCntr = Counter(objVar[right])
            leftSize, rightSize = sum(left), sum(right)
            leftWeight = calcEntropy(leftCntr) if self.asm else calcGini(leftCntr) # Calculating new Metric
            rightWeight = calcEntropy(rightCntr) if self.asm else calcGini(rightCntr)
            auxGain = (leftWeight*leftSize+rightWeight*rightSize)/len(catVar)
            if self.asm:
                auxGain -= totalVal
            if  (auxGain > bestMetric and self.asm) or (auxGain < bestMetric and not(self.asm)):
                # IF metric is best change best metric
                bestMetric = auxGain
                result[0] = c
                result[1] = bestMetric
        return result

    def bestNumSplit(self, numVar: np.array,objVar: np.array) -> tuple[float,float]:
        data = pd.DataFrame({'numVar':numVar,'objVar':objVar}).sort_values(by='numVar') # Sort the data
        rightCntr = defaultdict(int) # Create counters for calc Gini or Entropy
        bestMetric = 0
        best_split = numVar[0]
        rightCntr = Counter(objVar)
        totalVal = calcEntropy(rightCntr) if self.asm else calcGini(rightCntr)
        leftCntr = defaultdict(int)
        leftSize, rightSize = 0,len(numVar)
        for _, row in data.iterrows(): # Itering for every split
            rightCntr[row['objVar']] -= 1 # Changing the counters
            leftCntr[row['objVar']] += 1
            leftSize += 1
            rightSize -= 1
            if leftSize<self.minPt:
                continue
            elif rightSize < self.minPt:
                break
            leftWeight = calcEntropy(leftCntr) if self.asm else calcGini(leftCntr) # Calculating new Metric
            rightWeight = calcEntropy(rightCntr) if self.asm else calcGini(rightCntr)
            if self.asm:
                auxGain = totalVal-(leftWeight*leftSize+rightWeight*rightSize)/len(numVar)
            else:
                auxGain = (leftWeight*leftSize+rightWeight*rightSize)/len(numVar)
            if  (auxGain > bestMetric and self.asm) or (auxGain < bestMetric and not(self.asm)):
                # IF metric is best change best metric
                bestMetric = auxGain
                best_split = row['numVar']

        return best_split, bestMetric

    def findBestSplitter(self,X: np.array,y: np.array):
        totalMetrics = []
        for i in range(len(X[0])):
            column = X[:,i].flatten()
            if self.colTypes[i]=='cat':
                totalMetrics.append(self.bestCatSplit(column,y))
            else:
                totalMetrics.append(self.bestNumSplit(column,y,self.asm))
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

        bestVal = splitInfo[bestSplit] # The best Value
        ## Case for Basic Splits

        if self.colTypes[bestSplit] == 'cat': ## Binary Splitter
            actualNode = Node(lambda x: x[bestSplit]==bestVal, self._fit(X[(colInfo==bestVal)],y[(colInfo==bestVal)],depth+1), strCond=f'Column {bestSplit} == {bestVal}')
            oppositeNode = Node(lambda x: x[bestSplit]!=bestVal, self._fit(X[(colInfo!=bestVal)],y[(colInfo!=bestVal)],depth+1), strCond=f'Column {bestSplit} != {bestVal}')
        else:
            actualNode = Node(lambda x: x[bestSplit]<bestVal, self._fit(X[(colInfo<bestVal)],y[(colInfo<bestVal)],depth+1), strCond=f'Column {bestSplit} < {bestVal}')
            oppositeNode = Node(lambda x: x[bestSplit]>=bestVal, self._fit(X[(colInfo>=bestVal)],y[(colInfo>=bestVal)],depth+1), strCond=f'Column {bestSplit} >= {bestVal}')
            
        self.depthCount[depth] = self.depthCount.get(depth, 0)+2
        return [actualNode, oppositeNode]


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

    def _drawTree(self,node,ax,depth):

        xPos = (self.depthCount[depth]-self.depthCopy[depth]+1)*(self.depthAmpSize[depth])
        xPos += (self.depthCount[depth]-self.depthCopy[depth])*self.sqrSz
        self.depthCopy[depth]-=1
        yPos = depth*self.sqrSz+self.sqrSz//2*(depth+1)
        coords = (xPos, yPos)
        ax.add_patch(patches.Rectangle(coords,self.sqrSz,self.sqrSz,fill=True,color='blue' if isinstance(node,Node) else 'red'))
        ax.text(xPos+self.sqrSz//2,yPos+self.sqrSz//2,node.strCond,ha='center',va='center')

        if not isinstance(node,Leaf) and  not isinstance(node.next,Leaf):
            for n in node.next:
                newCoords = self._drawTree(n, ax, depth+1)
                ax.plot([xPos+self.sqrSz//2, newCoords[0]+self.sqrSz//2], [yPos+self.sqrSz, newCoords[1]], color='black')
        elif not isinstance(node, Leaf):
            newCoords = self._drawTree(node.next,ax,depth+1)
            ax.plot([xPos+self.sqrSz//2,newCoords[0]+self.sqrSz//2],[yPos+self.sqrSz,newCoords[1]],color='black')

        return xPos,yPos

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_point=3):
        self.mxDp = max_depth
        self.minPt = min_point

    def bestCatSplit(self ,catVar: np.array,objVar: np.array) -> tuple[str,float]:
        cntr = Counter(catVar)
        totalVal = mseMod(objVar) if self.asm else np.var(objVar) # Calc the initial Metric
        bestMetric = float('inf')
        result = [False,totalVal]   
        for c in cntr:
            left = catVar==c
            right = catVar!=c
            if sum(left) < self.minPt or sum(right) < self.minPt:
                continue
            leftWeight = mseMod(objVar[left]) if self.asm else np.var(objVar[left]) # Calculating new Metric
            rightWeight = mseMod(objVar[right]) if self.asm else np.var(objVar[right])
    
            if  leftWeight < totalVal and rightWeight < totalVal and np.mean([leftWeight,rightWeight])<bestMetric:
                # IF metric is best change best metric
                bestMetric = np.mean([leftWeight,rightWeight])
                result[0] = c
                result[1] = bestMetric
        return result

    def bestNumSplit(self, numVar: np.array,objVar: np.array) -> tuple[float,float]:
        data = pd.DataFrame({'numVar':numVar,'objVar':objVar}).sort_values(by='numVar') # Sort the data
        bestMetric = float('inf')
        best_split = numVar[0]
        totalVal = mseMod(objVar) if self.asm else np.var(objVar)

        for _, row in data.iterrows(): # Itering for every split
            left = data['numVar']<row['numVar']
            right = data['numVar']>=row['numVar']
            if sum(left)<self.minPt:
                continue
            elif sum(right) < self.minPt:
                break

            leftWeight = mseMod(objVar[left]) if self.asm else np.var(objVar[left]) # Calculating new Metric
            rightWeight = mseMod(objVar[right]) if self.asm else np.var(objVar[right])
    
            if  leftWeight < totalVal and rightWeight < totalVal and np.mean([leftWeight,rightWeight])<bestMetric:
                # IF metric is best change best metric
                bestMetric = np.mean([leftWeight,rightWeight])
                best_split = row['numVar']

        return best_split, bestMetric

    def findBestSplitter(self,X: np.array,y: np.array):
        totalMetrics = []
        for i in range(len(X[0])):
            column = X[:,i].flatten()
            if self.colTypes[i]=='cat':
                totalMetrics.append(self.bestCatSplit(column,y))
            else:
                totalMetrics.append(self.bestNumSplit(column,y,self.asm))
        return totalMetrics

    def _fit(self,X,y,depth = 0):
        if len(np.unique(y))==1:
            self.depthCount[depth] = self.depthCount.get(depth, 0)+1
            return Leaf(y[0])
        worstSplit = mseMod(y) if self.asm else np.var(y)
        splitters= self.findBestSplitter(X,y)
        splitInfo, splitValues = zip(*splitters)
        bestSplit = np.argmax(splitValues) if self.asm else np.argmin(splitValues)
        colInfo = X[:,bestSplit].flatten()

        ## Case when there worst splits
        if splitValues[bestSplit]==worstSplit or depth==self.mxDp:
            self.depthCount[depth] = self.depthCount.get(depth,0)+1
            self.depthCount[depth+1] = self.depthCount.get(depth+1, 0)+1
            return [Node(lambda x: True,Leaf(mode(y)),strCond='MaxDepth' if depth==self.mxDp else 'No Split')]

        bestVal = splitInfo[bestSplit] # The best Value
        ## Case for Basic Splits

        if self.colTypes[bestSplit] == 'cat': ## Binary Splitter
            actualNode = Node(lambda x: x[bestSplit]==bestVal, self._fit(X[(colInfo==bestVal)],y[(colInfo==bestVal)],depth+1), strCond=f'Column {bestSplit} == {bestVal}')
            oppositeNode = Node(lambda x: x[bestSplit]!=bestVal, self._fit(X[(colInfo!=bestVal)],y[(colInfo!=bestVal)],depth+1), strCond=f'Column {bestSplit} != {bestVal}')
        else:
            actualNode = Node(lambda x: x[bestSplit]<bestVal, self._fit(X[(colInfo<bestVal)],y[(colInfo<bestVal)],depth+1), strCond=f'Column {bestSplit} < {bestVal}')
            oppositeNode = Node(lambda x: x[bestSplit]>=bestVal, self._fit(X[(colInfo>=bestVal)],y[(colInfo>=bestVal)],depth+1), strCond=f'Column {bestSplit} >= {bestVal}')
            
        self.depthCount[depth] = self.depthCount.get(depth, 0)+2
        return [actualNode, oppositeNode]


    def createcolTypes(self,X):
        cols = []
        for i in range(len(X[0])):
            column = X[:,i].flatten()
            if np.issubdtype(column.dtype, np.int_) or np.issubdtype(column.dtype, np.str_):
                cols.append('cat')
            else:
                cols.append('num')
        self.colTypes = cols

    def fit(self,X,y, asm: str ='MSE'):
        # X categorical Features must be passed as objects or ints
        self.createcolTypes(X)
        self.asm = (asm == 'MSE')
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

    def _drawTree(self,node,ax,depth):

        xPos = (self.depthCount[depth]-self.depthCopy[depth]+1)*(self.depthAmpSize[depth])
        xPos += (self.depthCount[depth]-self.depthCopy[depth])*self.sqrSz
        self.depthCopy[depth]-=1
        yPos = depth*self.sqrSz+self.sqrSz//2*(depth+1)
        coords = (xPos, yPos)
        ax.add_patch(patches.Rectangle(coords,self.sqrSz,self.sqrSz,fill=True,color='blue' if isinstance(node,Node) else 'red'))
        ax.text(xPos+self.sqrSz//2,yPos+self.sqrSz//2,node.strCond,ha='center',va='center')

        if not isinstance(node,Leaf) and  not isinstance(node.next,Leaf):
            for n in node.next:
                newCoords = self._drawTree(n, ax, depth+1)
                ax.plot([xPos+self.sqrSz//2, newCoords[0]+self.sqrSz//2], [yPos+self.sqrSz, newCoords[1]], color='black')
        elif not isinstance(node, Leaf):
            newCoords = self._drawTree(node.next,ax,depth+1)
            ax.plot([xPos+self.sqrSz//2,newCoords[0]+self.sqrSz//2],[yPos+self.sqrSz,newCoords[1]],color='black')

        return xPos,yPos

# Prueba Clasificador Iris solo variables Númericas
myClassifier = DecisionTreeClassifier(max_depth=5)
myData = pd.read_csv('Iris.csv')
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

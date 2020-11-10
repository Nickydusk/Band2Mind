import numpy as np
import pandas as pd
import sklearn
import copy
from math import log2

class DecisionTree(object):
    def __init__(self):
        self.root = None

    def CART_chooseBestFeatureToSplit(self, X:np.ndarray, attrs):
        """
        bestFeature 有可能为-1， 表示找不到能够降低Entropy的属性，某种程度上起到了前剪枝的作用
        X: 输入数据集，最后一维为数据的标记。
        """
        baseEnt = self.calcEnt(X)
        bestInfoGainRatio = 0
        bestFeature = -1
        for i in range(X.shape[1]-1):
            if attrs[i] == False:
                continue
            values = set(X[:, i])
            newEnt = 0
            IV = 0
            for value in values:
                subDataset = X[X[:, i]==value]
                prop = subDataset.shape[0] / X.shape[0]
                newEnt += prop * self.calcEnt(subDataset)
                IV = IV - prop*log2(prop)
            infoGain = baseEnt - newEnt
            if (IV == 0):
                continue
            infoGainRatio = infoGain / IV
            if (infoGainRatio > bestInfoGainRatio):
                bestInfoGainRatio = infoGainRatio
                bestFeature = i
        return bestFeature

    def CART_createTree(self, X, attrs, node):
        tmp = X[:, attrs]
        allSame = True
        for i in range(tmp.shape[1]-1):
            if len(np.unique(tmp[:, i]))!=1:
                allSame = False
                break

        if np.unique(X[:, -1]).shape == (1,):
            node.toLeaf(X[0, -1])
        elif sum(attrs) <= 1 or allSame:
            node.toLeaf(self.majorityCount(X))
        else:
            bestFeature = self.CART_chooseBestFeatureToSplit(X, attrs)
            node.feature = bestFeature
            values = set(X[:, bestFeature])
            for value in values:
                X_v = X[X[:, bestFeature]==value]
                attrs_v = copy.deepcopy(attrs)
                attrs_v[bestFeature] = False
                newNode = TreeNode()
                node.childrens[value] = newNode
                self.CART_createTree(X_v, attrs_v, newNode)

    def calcEnt(self, X):
        tot = X.shape[0]
        values = set(X[:, -1])
        Ent = 0
        for value in values:
            num = X[X[:, -1] == value].shape[0]
            Ent -= (num/tot)*log2(num/tot)
        return Ent

    def splitDataset(self, X, i, value):
        return X[X[:, i]==value]

    def majorityCount(self, X):
        classCount = {}
        for vote in X[:, -1]:
            classCount[vote] = classCount.get(vote, 0) + 1
        sortedPairs = classCount.items()
        sortedPairs = sortedPairs.sort(key=lambda x:x[1], reverse=True)
        return sortedPairs[0][1]

    def train(self, X):
        attrs = [True for i in range(X.shape[1])]
        self.root = TreeNode()
        self.CART_createTree(X, attrs, self.root)

    def predict_sample(self, x):
        next = self.root
        while not next.isLeaf:
            feature = next.feature
            next = next.childrens.get(x[feature])
        return next.label

    def predict_batch(self, X):
        return np.asarray(list(map(lambda x:self.predict_sample(x), X)))



class TreeNode(object):
    def __init__(self, feature=None, isLeaf=False, label=None):
        self.feature = feature
        self.isLeaf = isLeaf
        self.label = label
        self.childrens = {}
    
    def toLeaf(self, label):
        self.isLeaf = True
        self.label = label

if __name__ == "__main__":
    data = pd.read_excel("./data.xslx").values[:, 1:]
    agent = DecisionTree()
    agent.train(data)
    res = agent.predict_batch(data)
    print(res)
import numpy as np
from numpy import e, dot, log
from . import Hyperparameters
from . import datasetmethods as dsmethods
#from sklearn import datasets
#import pandas as pd

class LogisticRegression():
    def fit(self, X, y, hp : Hyperparameters):
        training, testing = dsmethods.trainTestSplit([X, y], hp.trP, hp.tstP)

        self.train(training[0], training[1], hp)
        results = self.test(testing[0], testing[1])

        return results
    
    def test(self, X, y):
        stats = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        for i in range(X.shape[1]):
            yPred = 1 if self._sigmoid(dot(X[i,:], self.W)) > 0.5 else 0
            if yPred == 1:
                if y[i] == 1:
                    stats["TP"] += 1
                else:
                    stats["FP"] += 1
            else:
                if y[i] == 1:
                    stats["FN"] += 1
                else: 
                    stats["TP"] += 1
        
        stats["accuracy"] = (stats["TP"]+stats["TN"])/len(y)
        stats["precision"] = (stats["TP"])/(stats["TP"] + stats["FP"])
        stats["sensitivity"] = (stats["TP"])/(stats["TP"] + stats["FN"])
        stats["specificity"] = stats["TN"]/(stats["TN"] + stats["FP"])
        stats["f1score"] = (2*stats["precision"]*stats["sensitivity"])/(stats["precision"]+stats["sensitivity"])

        return stats

    def train(self, X, y, hyPm : Hyperparameters):
        N = len(X)
        self.loss = np.zeros(hyPm.epochs)

        self.W = np.random.rand(X.shape[1])

        for i in range(hyPm.epochs):
            # get predicted y-value given inputs and Weight
            yPred = self.sigmoid(dot(X, self.W))
            # gradient descent step
            self.W -= hyPm.lr * dot(X.T, yPred - y)/N
            self.loss[i] = self._loss(X, y)

    def predict(self, X):
        z = dot(X, self.W)
        return [1 if i > 0.5 else 0 for i in self._sigmoid(z)]
        

    def _loss(self, X, y):
        z = dot(X, self.W)
        cls1 = y * log(self._sigmoid(z))
        cls0 = (1-y)*log(1-self._sigmoid(z))

        l = -sum(cls1 + cls0)/self.N
        return l
    
    def _sigmoid(self, z):
        return 1/(1+e**(-z))
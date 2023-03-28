import numpy as np
import Hyperparameters
from sklearn.datasets import make_classification
import datasetmethods as dsmethods

# from sklearn import datasets
# import pandas as pd


class LogisticRegression:
    def fit(self, X, y, hp: Hyperparameters.Hyperparameters):
        training, testing = dsmethods.trainTestSplit([X, y], hp.trP, hp.tstP)
        
        self.train(training[0], training[1], hp)
        results = self.test(testing[0], testing[1])

        return results

    def test(self, X, y):
        stats = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        for i in range(X.shape[0]):
            yPred = 1 if self._sigmoid(np.dot(X[i, :], self.W)) > 0.5 else 0
            if yPred == 1:
                if y[i] == 1:
                    stats["TP"] += 1
                else:
                    stats["FP"] += 1
            else:
                if y[i] == 1:
                    stats["FN"] += 1
                else:
                    stats["TN"] += 1

        stats["accuracy"] = (stats["TP"] + stats["TN"]) / len(y)
        stats["precision"] = (stats["TP"]) / (stats["TP"] + stats["FP"])
        stats["sensitivity"] = (stats["TP"]) / (stats["TP"] + stats["FN"])
        stats["specificity"] = stats["TN"] / (stats["TN"] + stats["FP"])
        stats["f1score"] = (2 * stats["precision"] * stats["sensitivity"]) / (
            stats["precision"] + stats["sensitivity"]
        )

        return stats

    def train(self, X, y, hyPm: Hyperparameters.Hyperparameters):
        '''
        X -> input data
        y -> actual output
        hyPm -> hyperparameters object
        '''
        N, Nf = X.shape
        y = y.reshape(1, N)
        self.loss = np.zeros(hyPm.epochs)
        
        self.W = np.random.rand(Nf)
        self.bias = 0

        for e in range(hyPm.epochs):
            for i in range((N-1)//hyPm.bs+1):
                XBatch = X[i*hyPm.bs:(i+1)*hyPm.bs]
                yBatch = y[0, i*hyPm.bs:(i+1)*hyPm.bs]#.reshape(hyPm.bs, 1)
                # get predicted y-value given inputs and Weight
                yPred = self._sigmoid(np.dot(XBatch, self.W) + self.bias)
                # gradient descent step
                self.W -= hyPm.lr*(1/N)*np.dot(XBatch.T, (yPred - yBatch))
                self.bias -= hyPm.lr*(1/N)*np.sum((yPred - yBatch))
            
            self.loss[e] = self._loss(X, y)
            

    def predict(self, X):
        z = np.dot(X, self.W)
        yPred = self._sigmoid(z)
        return [1 if i >= 0.5 else 0 for i in yPred]

    def _loss(self, X, y):
        z = np.dot(X, self.W) + self.bias
        cls1 = y * np.log(self._sigmoid(z))
        cls0 = (1 - y) * np.log(1 - self._sigmoid(z))

        l = -np.sum(cls1 + cls0) / X.shape[0]
        return l

    def _sigmoid(self, z):
        return 1 / (1 + np.e**(-z))

def test(n, nf, hp, verbose=True):
    X, y = make_classification(
        n_samples=n,
        n_features=nf,
    )

    logit = LogisticRegression()
    results = logit.fit(X, y, hp)
    if verbose:
        for k, v in results.items():
            print(k + ": " + str(v))

    return results

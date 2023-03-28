import numpy as np


class NaiveBayes:
    def fit(self, X, y, type: str = "Gaussian"):
        match (type.capitalize):
            case "GAUSSIAN", "GAUSS", "NORMAL", "NORM", "G", "N":
                pass
            case "BERNOULLI", "BERN", "B":
                pass
            case "MULTINOMIAL", "MULTI", "MULT", "M":
                pass

    def train(self, X, y):
        pass

    def predict():
        pass

    def _gaussian():
        pass

    def _bernoulli():
        pass

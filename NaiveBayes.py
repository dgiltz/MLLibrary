import numpy as np
from Hyperparameters import Hyperparameters
import datasetmethods as dsmethods

class NaiveBayes:
    def fit(self, X : np.array, y : np.array, hp : Hyperparameters, eventModel : str = "Guassian"):
        train, test = dsmethods.trainTestSplit([X, y], hp.trP, hp.tstP)
        num_cls = np.max(y) + 1
        num_feat = X.shape[1]

        match (eventModel.capitalize):
            case "GAUSSIAN", "GAUSS", "NORMAL", "NORM", "G", "N":
                X_train, y_train = train
                
            case "BERNOULLI", "BERN", "B":
                pass
            case "MULTINOMIAL", "MULTI", "MULT", "M":
                pass

    def train(self, X, y):
        pass
    
    def test(self, X, y):
        stats = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        for i in range(X.shape[0]):
            yPred = None 
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
    
    def predict():
        pass
    
    def _gaussian(self, x, mean, var):
        c = 1/(2*np.pi*var)
        exp = np.exp(-(x-mean)^2/(2*var))
        return c*exp

    def _get_mean_variance(self, X, y):
        segmented = self._segment_by_class(X, y)
        num_feat = X.shape[1]

        mean_var_num = {}
        for cls in segmented.keys():
            for feat in num_feat:
                mean_var_num[cls][feat]["mean"] = np.mean(segmented[cls][:, feat])
                mean_var_num[cls][feat]["var"] = np.var(segmented[cls][:, feat])
                mean_var_num[cls]["num"] = segmented[cls][:, feat].shape[0]
                
        return mean_var_num

    def _segment_by_class(self, X, y):
        segmented =  {}
        for row in range(X.shape[0]):
            if y[row] in segmented.keys():
                segmented[y[row]] = np.row_stack(X[row, :], segmented[y[row]])  
            else:
                segmented[y[row]] = X[row, :]
        
        return segmented
    
    def _get_prior(self, y, num_cls, type: str = "default"):
        prior = np.ones(num_cls)
        match(type.capitalize):
            case "MLE", "MAXIMUM LIKELIHOOD", "MAXIMUM LIKELIHOOD ESTIMATION":
                prior -= 1
                for cls in y:
                    prior[cls] += 1
            case "DEFAULT":
                pass
                
        return prior/num_cls


    def _bernoulli(self, x, p):
        pass

    def _multinomial(self, x, mp):
        pass

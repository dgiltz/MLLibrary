import numpy as np
from numpy import random
import pandas as pd

class NaiveBayes():
    def fit(self, X, y, type : str = "Gaussian"):
        match(type.capitalize):
            case "GAUSSIAN","GAUSS","NORMAL","NORM","G","N":
                pass
            case "BERNOULLI","BERN","B":
                pass
            case "MULTINOMIAL", "MULTI", "MULT", "M":
                pass
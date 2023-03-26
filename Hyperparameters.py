from . import LogisticRegression, modeltypes
class Hyperparameters:
    def __init__(self):
        self.lr = 0.0
        self.bs = 0.0
        self.epochs = 0.0
        self.trP = 0.0
        self.tstP= 0.0
        self.valdSplit = 0.0
    
    def setDefault(self, model : modeltypes):
        if type(model) == LogisticRegression:
            self.lr = 1e-3
            self.epochs = 4
            self.trP = 0.9
            self.tstP = 0.1

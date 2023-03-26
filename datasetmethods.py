import numpy as np
from numpy import concatenate as concat
from numpy.random import shuffle as npshuffle

def trainTestSplit(arrays, trP, tstP, shuffle=True):
    '''
    arrays -> a list like collection of numpy matrices and arrays with the same length (first dim) that represent the dataset
    trP : float -> percentage of dataset to be used for training, between 0.0 and 1.0
    tstP : float -> percentage of dataset to be used for testing, between 0.0 and 1.0
    shuffle : bool -> wether to shuffle array order to (generally) ensure more complete randomness and reduce bias
    '''
    if len(arrays) == 0:
        raise ValueError("Need at least one array as input")
    
    for i in range(len(arrays)-1):
        if arrays[i].shape[0] != arrays[i+1].shape[0]:
            raise ValueError("All arrays must have the same amount of rows")
    
    sizes = [len(arrays[i]) for i in range(len(arrays))]

    data = concat(arrays, axis=0)
    if shuffle:
        data = npshuffle(data)

    N = arrays[i].shape[0]
    trSize = round(N*trP)
    tstSize = round(N*tstP)
    if trSize + tstSize > N:
        raise ValueError("Training and Test Split invalid, must add to at most 1.0")
    
    elif trP + tstP == 1.0 and trSize + tstSize < N:
        tstSize = N-trSize

    trData = data[:trSize] ; tstData = data[trSize:tstSize]

    training = np.array(np.empty((1, sizes[i])) for i in range(len(arrays)))
    testing = np.array(np.empty((1, sizes[i])) for i in range(len(arrays)))
    for sz in sizes:
        prev = 0
        for i in range(trSize):
            training[sz] = concat((training[sz], trData[i, prev:prev+sz]), axis=1)
        
        for j in range(tstSize):
            testing[sz] = concat((testing[sz], tstData[j, prev:prev+sz]), axis=1)
        
        prev += sz

    return training, testing




    




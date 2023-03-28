import numpy as np
from numpy import column_stack, row_stack
from numpy.random import shuffle as npshuffle


def trainTestSplit(arrays, trP, tstP, shuffle=True):
    """
    arrays -> a list like collection of numpy matrices and arrays with the same length (first dim) that represent the dataset
    trP : float -> percentage of dataset to be used for training, between 0.0 and 1.0
    tstP : float -> percentage of dataset to be used for testing, between 0.0 and 1.0
    shuffle : bool -> wether to shuffle array order to (generally) ensure more complete randomness and reduce bias
    """
    if len(arrays) == 0:
        raise ValueError("Need at least one array as input")

    for i in range(len(arrays) - 1):
        if arrays[i].shape[0] != arrays[i + 1].shape[0]:
            raise ValueError("All arrays must have the same amount of rows")

    sizes = []
    for arr in arrays:
        if len(arr.shape) == 1:
            sizes.append(1)
        else:
            sizes.append(arr.shape[1])

    N = arrays[i].shape[0]

    trSize = round(N * trP)
    tstSize = round(N * tstP)
    if trSize + tstSize > N:
        raise ValueError("Training and Test Split invalid, must add to at most 1.0")

    elif trP + tstP == 1.0 and trSize + tstSize < N:
        tstSize = N - trSize

    training = []
    testing = []

    if shuffle:
        data = column_stack(arrays)
        npshuffle(data)

        trData = data[:trSize, :]
        tstData = data[trSize : trSize + tstSize, :]

        prev = 0
        for sz in sizes:
            training.append(trData[:, prev : prev + sz])
            testing.append(tstData[:, prev : prev + sz])
            prev += sz

    else:
        for arr in arrays:
            training.append(arr[:trSize])
            testing.append(arr[trSize:tstSize])

    return training, testing


def normalize(data: np.array, type: str = "ZSCORE"):
    match (type.capitalize):
        case "RANGE", "RNG", "R":
            return (data - np.min(data)) / (np.max(data) - np.min(data))
        case "MEANSTD", "ZSCORE", "Z":
            return (data - np.mean(data)) / (np.std(data))
        case "LOG", "LOGARITHM", "L":
            return np.log(data)

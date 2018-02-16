import pickle # used for deserializing the already packed input data
import gzip   # input data in the gz compression; decompress in order to obtain read permission
import numpy as np # ordinary linear algebra library

def loadMnist():
    file = gzip.open('mnist_data/mnist.pkl.gz', 'rb') # rb: read-binary
    # decompress in 3 pairs, p[0] is a matrix[n*784] representing the unraveled images (28*28 = 784), p[1] is a vector[n] containing the exact label for each p[0] row
    train, val, test = pickle.load(file, encoding="latin1")
    file.close()
    # divide each pair in 2 separate lists and combine them, this will becomes handy for later work, where the pairs are needed to be looped together
    trainInputs = [np.reshape(img,(784,1)) for img in train[0]]
    trainLabels = [vectorizeLabel(label) for label in train[1]]
    train = list(zip(trainInputs, trainLabels))
    valInputs = [np.reshape(x,(784,1)) for x in val[0]]
    val = list(zip(valInputs, val[1]))
    testInputs = [np.reshape(x,(784,1)) for x in test[0]]
    test = list(zip(testInputs,test[1]))
    return (train, val, test)

def vectorizeLabel(j):
    # build and return the 10 dimension unit vector associated with the exact label, handy for the backpropagtion algorithm
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
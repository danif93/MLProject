
# coding: utf-8

# In[ ]:


import mnist_loader as load
import neural_network as neuronet
import activation_functions as af
import optimization_functions as of
import cost_functions as cf
import learningStep_generators as lg

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[ ]:


# parameters initialization

maxEpochs = 50              # the neural net will not train over maxEpochs
earlyStopParameter = 5      # number of last error entries the net will take in consideration for early stopping
earlyStopThreshold = 0.14   # errors-vector norm threshold for deciding early stop

batchSize = 10              # stochastic gradient descent batch size
regularParam = 3.0          # lambda
hiddenUnit = 20             # neurons number for the single hidden unit layer

decayStep = lg.harmonicSeries(3) # at each epoch the step will shrink over the harmonic series
constStep = lg.constant(3)       # at each epoch the step will remain constant

weightInitErr = 0.3         # if requested, a low delta for weights initialization

tr, va, te = load.loadMnist() # train, validation and test datasets


# In[ ]:


# Simple execution

sigmoidNetCross = [neuronet.NeuralNetwork([784, hiddenUnit, 10], unit=af.Sigmoid, cost=cf.MeanSquareError)]
sgdC = of.StochasticGradientDescent(sigmoidNetCross)
validAccuracy, trainAccuracy, testAccuracy = sgdC.SGD(tr, va, te, maxEpochs, batchSize, decayStep, earlyStopParameter, earlyStopThreshold)


# In[ ]:


# Constant - Decaying step comparison
# considerations about oscillating, saturation, early stopping and learning break

sigmoidNetCross = [neuronet.NeuralNetwork([784, hiddenUnit, 10], unit=af.Sigmoid, cost=cf.CrossEntropy)]
sgdC = of.StochasticGradientDescent(sigmoidNetCross)
validAccuracyC, trainAccuracyC, testAccuracyC = sgdC.SGD(tr, va, te, maxEpochs, batchSize, constStep, earlyStopParameter, earlyStopThreshold)
sigmoidNetCross = [neuronet.NeuralNetwork(net=[784, hiddenUnit, 10], unit=af.Sigmoid, cost=cf.CrossEntropy)]
sgdC = of.StochasticGradientDescent(sigmoidNetCross)
validAccuracyD, trainAccuracyD, testAccuracyD = sgdC.SGD(tr, va, te, maxEpochs, batchSize, decayStep, earlyStopParameter, earlyStopThreshold)

nonZeroConstant = len(np.argwhere(validAccuracyC[0]))
nonZeroDecaying = len(np.argwhere(validAccuracyD[0]))

plt.title('Constant/Decaying learning step')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([80,100])
plt.plot(np.arange(nonZeroConstant),validAccuracyC[0][:nonZeroConstant], label='Costant')
plt.plot(nonZeroConstant, testAccuracyC[0], 'x', label='Costant - Test')
plt.plot(np.arange(nonZeroDecaying),validAccuracyD[0][:nonZeroDecaying], label='Harmonic Decaying')
plt.plot(nonZeroDecaying, testAccuracyD[0], 'x', label='Harmonic Decaying - Test')
plt.grid()
plt.legend()


# In[ ]:


# Cross Entropy - MSE comparison

sigmoidNetCross = [neuronet.NeuralNetwork([784, hiddenUnit, 10], unit=af.Sigmoid, cost=cf.CrossEntropy, weightError=weightInitErr)]
sigmoidNetMSE = [neuronet.NeuralNetwork(net=[784, hiddenUnit, 10], unit=af.Sigmoid, cost=cf.MeanSquareError, weightError=weightInitErr)]
sgdC = of.StochasticGradientDescent(sigmoidNetCross)
sgdM = of.StochasticGradientDescent(sigmoidNetMSE)

validAccuracyM, trainAccuracyM, testAccuracyM = sgdM.SGD(tr, va, te, maxEpochs, batchSize, decayStep, earlyStopParameter, earlyStopThreshold)
validAccuracyC, trainAccuracyC, testAccuracyC = sgdC.SGD(tr, va, te, maxEpochs, batchSize, decayStep, earlyStopParameter, earlyStopThreshold)

nonZeroCross = len(np.argwhere(validAccuracyC[0]))
nonZeroMSE = len(np.argwhere(validAccuracyM[0]))

plt.title('Cross-Entropy / MSE training result')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0,100])
plt.plot(np.arange(nonZeroCross), validAccuracyC[0][:nonZeroCross], label='Cross-Entropy Validation')
plt.plot(nonZeroCross, testAccuracyC[0], 'x', label='Cross-Entropy - Test')
plt.plot(np.arange(nonZeroMSE), validAccuracyM[0][:nonZeroMSE], label='MSE')
plt.plot(nonZeroMSE, testAccuracyM[0], 'x', label='MSE - Test')

plt.grid()
plt.legend()


# In[ ]:


# Training - Validation accuracy comparison 

#SGD(trainingSet, validSet, testSet, numEpochs, batchSize, stepGen, earlyStopParam, earlyStopThrshld, _lambda=0.0, l2Regul=True, trainEval=False)
sigmoidNetCross = [neuronet.NeuralNetwork([784, hiddenUnit, 10], unit=af.Sigmoid, cost=cf.CrossEntropy)]
sgdC = of.StochasticGradientDescent(sigmoidNetCross)
validAccuracy, trainAccuracy, testAccuracy = sgdC.SGD(tr, va, te, maxEpochs, batchSize, decayStep, earlyStopParameter, earlyStopThreshold, trainEval=True)

nonZeroValues = len(np.argwhere(validAccuracy[0]))

plt.title('Training/Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([80,100])
plt.plot(np.arange(nonZeroValues), validAccuracy[0][:nonZeroValues], label='Validation Accuracy')
plt.plot(np.arange(nonZeroValues), trainAccuracy[0][:nonZeroValues], label='Train Accuracy')
plt.plot(nonZeroValues, testAccuracy[0], 'x', label='Test Accuracy')
plt.legend()
plt.grid()


# In[ ]:


# Training - Validation accuracy comparison with l2 regularization

sigmoidNetCross = [neuronet.NeuralNetwork([784, hiddenUnit, 10], unit=af.Sigmoid, cost=cf.CrossEntropy)]
sgdC = of.StochasticGradientDescent(sigmoidNetCross)
validAccuracy, trainAccuracy, testAccuracy = sgdC.SGD(tr, va, te, maxEpochs, batchSize, decayStep, earlyStopParameter, earlyStopThreshold, _lambda=regularParam, trainEval=True)

nonZeroValues = len(np.argwhere(validAccuracy[0]))

plt.title('Training/Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([80,100])
plt.plot(np.arange(nonZeroValues), validAccuracy[0][:nonZeroValues], label='Validation Accuracy')
plt.plot(np.arange(nonZeroValues), trainAccuracy[0][:nonZeroValues], label='Train Accuracy')
plt.plot(nonZeroValues, testAccuracy[0], 'x', label='Test Accuracy')
plt.legend()
plt.grid()


# In[ ]:


# Training - Test accuracy comparison with l1 regularization

sigmoidNetCross = [neuronet.NeuralNetwork([784, hiddenUnit, 10], unit=af.Sigmoid, cost=cf.CrossEntropy)]
sgdC = of.StochasticGradientDescent(sigmoidNetCross)
validAccuracy, trainAccuracy, testAccuracy = sgdC.SGD(tr, va, te, maxEpochs, batchSize, decayStep, earlyStopParameter, earlyStopThreshold, _lambda=regularParam, l2Regul=False, trainEval=True)

nonZeroValues = len(np.argwhere(validAccuracy[0]))

plt.title('Training/Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([80,100])
plt.plot(np.arange(nonZeroValues), validAccuracy[0][:nonZeroValues], label='Validation Accuracy')
plt.plot(np.arange(nonZeroValues), trainAccuracy[0][:nonZeroValues], label='Train Accuracy')
plt.plot(nonZeroValues, testAccuracy[0], 'x', label='Test Accuracy')
plt.legend()
plt.grid()


# In[ ]:


# hold-out validation on hidden unit number

nets = [neuronet.NeuralNetwork([784, hiddUnitNumber, 10], unit=af.Sigmoid, cost=cf.CrossEntropy) for hiddUnitNumber in [50,100,150,200]]
sgdC = of.StochasticGradientDescent(nets)
validAccuracy, trainAccuracy, testAccuracy = sgdC.SGD(tr, va, te, maxEpochs, batchSize, decayStep, earlyStopParameter, earlyStopThreshold, _lambda=regularParam)

plt.title('Validation accuracy per hidden unit model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([90,100])
for i in np.arange(len(nets)):
    nonZeroValues = len(np.argwhere(validAccuracy[i]))
    plt.plot(np.arange(nonZeroValues),validAccuracy[i][:nonZeroValues], label='{}'.format(nets[i].net[1:-1]))
    plt.plot(nonZeroValues, testAccuracy[i], 'x', label='{} - Test'.format(nets[i].net[1:-1]))    
plt.legend()
plt.grid()


# In[ ]:


# AdaGrad algo - not working

sigmoidNetCross = neuronet.NeuralNetwork(net=[784, hiddenUnit, 10], unit=af.Sigmoid, cost=cf.CrossEntropy)
agC = of.AdaGrad(sigmoidNetCross)
testAccuracy, trainAccuracy = agC.AG(tr, va, te, maxEpochs, batchSize, constStep, _lambda=regularParam, trainEval=True)


plt.title('Training/Test accuracy ')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([80,100])
plt.plot(np.arange(epochs), testAccuracy, label='Test Accuracy')
plt.plot(np.arange(epochs), trainAccuracy, label='Train Accuracy')
plt.legend()
plt.grid()


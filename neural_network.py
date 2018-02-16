import numpy as np  # ordinary linear algebra library

class NeuralNetwork(object):
    
    # constructor: takes as input an array describing the network with the chosen activation unit and cost function
    # e.g. [3,2,1] is a network with 3 input, 2 hidden and 1 output neurons
    def __init__(self, net, unit, cost, weightError=0.0):
        
        # chosen activation unit function initialization
        self.activFun = unit.activationFunction
        self.derActivFun = unit.derivativeFunction
        
        # cost function
        self.computeError = cost.computeError
        
        # net utilities
        self.layers = len(net)
        self.net = net
        
        # build a list of vector, one for each layer; the lenght of each vector is equal to the quantity of neurons in that layer, and store under the i-th cell the bias for the i-th neuron in that layer
        self.biasesLayers = [np.abs(np.ones((neurNum, 1))*0.1) for neurNum in net[1:]]
        # build a list of matrices, one for each set of edges bridging two layers; the index of each matrix are swapped in order to perform correctly the dot-product of the activation computation and ignoring all the transpose operations.
        self.weightsLayers = [np.random.randn(output, _input)+weightError for _input, output in zip(net[:-1], net[1:])]
        

    def backpropagation(self, img, label):
        biasesDeriv = [np.zeros(biasLayer.shape) for biasLayer in self.biasesLayers]
        weightsDeriv = [np.zeros(weightLayer.shape) for weightLayer in self.weightsLayers]
        
        # PHASE 1: feedforward
        activation = img    # first activation: input image
        activations = [img] # layer by layer activations
        hidUnitInputs = []  # layer by layer linear transformation
        # for each net layer:
        for biasLayer, weightLayer in zip(self.biasesLayers, self.weightsLayers):
            hui = np.dot(weightLayer, activation)+biasLayer # w*input_act+b
            hidUnitInputs.append(hui)
            activation = self.activFun(hui) # output activation
            activations.append(activation)
            
        # PHASE 2: backpropagation
        # compute the error for the last layer with the chosen loss function
        error = self.computeError(label, activation, self.derActivFun, hui) 
        biasesDeriv[-1] = error                                       # bias_deriv=layer_error
        weightsDeriv[-1] = np.dot(error, activations[-2].transpose()) # weight_deriv=(layer_error*input_activ_transp)
        # backward step in the net:
        for layer in range(2, self.layers):
            hui = hidUnitInputs[-layer]
            error = np.dot(self.weightsLayers[-layer+1].transpose(), error) * self.derActivFun(hui) # error=(prev_weight_transp*prev_error)*saturation
            biasesDeriv[-layer] = error
            weightsDeriv[-layer] = np.dot(error, activations[-layer-1].transpose())
        return (biasesDeriv, weightsDeriv)
    
                  
    # network feed process, apply the activation function on top of the linear affine transformation: activFun(w*a+b)
    def feedForward(self, inputImg):
        activation=inputImg
        for biasLayer, weightLayer in zip(self.biasesLayers, self.weightsLayers):
            activation = self.activFun(np.dot(weightLayer,activation)+biasLayer)
        return activation
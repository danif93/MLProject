import numpy as np

class Sigmoid:        
    def activationFunction(z):
        return 1.0/(1.0+np.exp(-z))
    def derivativeFunction(z):
        return Sigmoid.activationFunction(z)*(1-Sigmoid.activationFunction(z))
    
class AbsoluteValueReLU:
    def activationFunction(z):
        return np.abs(z)
    def derivativeFunction(z):
        return z/np.abs(z)
    
class HyperbolicTangent:
    def activationFunction(z):
        return np.tanh(z)
    def derivativeFunction(z):
        return 1-(np.tanh(z)**2)
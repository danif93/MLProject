import numpy as np

class MeanSquareError:
    def lossFunction(label, outActiv):
        return ((np.linalg.norm(label-outActiv)**2)*0.5)
    def computeError(label, outActiv, derActivFun, wInput):
        return (outActiv-label)*derActivFun(wInput)
    
class CrossEntropy:
    def lossFunction(label, outActiv):
        # np.nan_to_num converge ln(\approx 0) to a number
        return np.sum(np.nan_to_num(-label*np.log(outActiv)-(1-label)*np.log(1-outActiv)))
    def computeError(label, outActiv, derActivFun, wInput):
        return (outActiv-label)
        
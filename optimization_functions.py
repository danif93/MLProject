import numpy as np
import random       # used for shuffling the list containing the training data
                    # for the stochastic gradient descent

# compute the number of correct guesses on a test set
def evaluate(testSet, neuralNet):
        testSize = len(testSet)
        results = [(np.argmax(neuralNet.feedForward(img)), np.argmax(label)) for (img,label) in testSet] if testSize>10000 else [(np.argmax(neuralNet.feedForward(img)), label) for (img,label) in testSet]
        successes = sum(int(computed==expected) for (computed,expected) in results)
        ratio = successes*100/testSize
        print("{} / {} ~= {}%".format(successes, testSize, ratio));
        return ratio          


class StochasticGradientDescent(object):
    def __init__(self, nets):
        self.neuroNets = nets
        
    def SGD(self, trainingSet, validSet, testSet, numEpochs, batchSize, stepGen, earlyStopParam, earlyStopThrshld, _lambda=0.0, l2Regul=True, trainEval=False):
        
        trainSize=len(trainingSet)
        successEvalValid = np.zeros((len(self.neuroNets),numEpochs)) # contains the validation accuracy values
        successEvalTrain = np.zeros((len(self.neuroNets),numEpochs)) # contains the training accuracy values
        successEvalTest = np.zeros(len(self.neuroNets))              # contains the test error values
        normLambda = _lambda/trainSize
        
        i = 0   # nets cycle index        
        for net in self.neuroNets:
            print("net {} with {} hidden neurons".format(i, net.net[1:-1]))
            epoch=0
            earlyStop=np.zeros(earlyStopParam) # store the validation accuracy for the last earlyStopParam epochs
            while True: # until reach maxEpochs or the net stops learning
                step = next(stepGen)
                random.shuffle(trainingSet)
                # train with every batch
                for k in range(0, trainSize, batchSize):
                    self.updateNetwork(i, trainingSet[k:k+batchSize], step, normLambda, l2Regul)
                earlyStop[epoch%earlyStopParam] = evaluate(validSet, net)  # validation accuracy
                successEvalValid[i,epoch] = earlyStop[epoch%earlyStopParam]
                if trainEval:
                    successEvalTrain[i,epoch] = evaluate(trainingSet, net) # training accuracy
                lastPredictionsErrors = np.linalg.norm(earlyStop-successEvalValid[i,epoch]) # norm for the last earlyStopParam epochs
                if (lastPredictionsErrors <= earlyStopThrshld) or (epoch == numEpochs-1):
                    print("stopped at epoch {}".format(epoch))
                    break
                epoch += 1
                #endwhile - epoch cycle
            print("Error on test set for net {} with {} hidden neurons:".format(i,net.net[1:-1]))
            successEvalTest[i] = evaluate(testSet, net) # test error
            i += 1
            #endfor - nets cycle
        selected = np.argmax(successEvalTest) # find the net index with the lowest test error and possibly do some stuff...
        return (successEvalValid, successEvalTrain, successEvalTest)
    
    # update network biases and weights training on a selected set
    def updateNetwork(self, i, trainingSet, step, normLambda, l2Regul):
        normStep = step/len(trainingSet) # normalized since it is requested to compute the average between all the derivatives for a single img
        # initialize the derivation lists for the biases and weights
        sumBiasesDeriv = [np.zeros(biasLayer.shape) for biasLayer in self.neuroNets[i].biasesLayers]
        sumWeightsDeriv = [np.zeros(weightLayer.shape) for weightLayer in self.neuroNets[i].weightsLayers]
        
        # train with the given set: compute the derivative for weights and biases with backpropagation algo and sum them up
        for img, label in trainingSet:
            biasesDeriv, weightsDeriv = self.neuroNets[i].backpropagation(img, label)
            sumBiasesDeriv = [newBD+sumBD for newBD, sumBD in zip(biasesDeriv, sumBiasesDeriv)]
            sumWeightsDeriv = [newWD+sumWD for newWD, sumWD in zip(weightsDeriv, sumWeightsDeriv)]
        
        # update the network biases and weights following the equation: x'=x-delta(x) where delta(x)=step*gradient(costFun(x)) and x=(w,b)
        self.neuroNets[i].biasesLayers = [oldB-(normStep*newB) for oldB, newB in zip(self.neuroNets[i].biasesLayers, sumBiasesDeriv)]
        # choice between l2 or l1 regul based on the flag l2Regul
        self.neuroNets[i].weightsLayers = [(1-(step*normLambda))*oldW-(normStep*newW) for oldW, newW in zip(self.neuroNets[i].weightsLayers, sumWeightsDeriv)] if l2Regul else [oldW-(step*normLambda*np.sign(oldW))-(normStep*newW) for oldW, newW in zip(self.neuroNets[i].weightsLayers, sumWeightsDeriv)]


        
        
        
class AdaGrad(object):
    def __init__(self, net):
        self.neuralNet = net
    
    def AG(self, trainingSet, numEpochs, batchSize, step, _lambda=0.0, l2Regul=True, testSet=None):
        trainSize=len(trainingSet)
        successEvalTest = []
        successEvalTrain = []
        normLambda = _lambda/trainSize
        for epoch in range(numEpochs):
            random.shuffle(trainingSet)
            # train with every batch
            for k in range(0, trainSize, batchSize):
                self.updateNetwork(trainingSet[k:k+batchSize], step, normLambda, l2Regul)
            if testSet: 
                successEvalTest.append(evaluate(testSet, self.neuralNet))
                #uncomment below if computing accuracy ratio between test-train
                #successEvalTrain.append(evaluate(trainingSet, self.neuralNet))
            else: 
                print("Epoch {} trained".format(epoch))
        return None if testSet==None else (successEvalTest,successEvalTrain)
    
    # update network biases and weights training on a selected set
    def updateNetwork(self, trainingSet, step, normLambda, l2Regul):
        normStep = step/len(trainingSet) # normalized since it is requested to compute the average between all the derivatives for a single img
        # initialize the derivation lists for the biases and weights
        sumBiasesDeriv = [np.zeros(biasLayer.shape) for biasLayer in self.neuralNet.biasesLayers]
        sumWeightsDeriv = [np.zeros(weightLayer.shape) for weightLayer in self.neuralNet.weightsLayers]
        
        # train with the given set: compute the derivative for weights and biases with backpropagation algo and sum them up
        for img, label in trainingSet:
            biasesDeriv, weightsDeriv = self.neuralNet.backpropagation(img, label)
            sumBiasesDeriv = [np.multiply(newBD,newBD)+sumBD for newBD, sumBD in zip(biasesDeriv, sumBiasesDeriv)]
            sumWeightsDeriv = [np.multiply(newWD,newWD)+sumWD for newWD, sumWD in zip(weightsDeriv, sumWeightsDeriv)]
            deltab=[-1*(np.divide(normStep,(10**-7+np.sqrt(r))))*newBD for newBD, r in zip(biasesDeriv, sumBiasesDeriv)]
            deltaw=[-1*(np.divide(normStep,(10**-7+np.sqrt(r))))*newWD for newWD, r in zip(weightsDeriv, sumWeightsDeriv)]
        
        # update the network biases and weights following the equation: x'=x-delta(x) where delta(x)=step*gradient(costFun(x)) and x=(w,b)
        self.neuralNet.biasesLayers = [oldB+newB for oldB, newB in zip(self.neuralNet.biasesLayers, sumBiasesDeriv)]
        # choice between l2 or l1 regul based on the flag l2Regul
        self.neuralNet.weightsLayers = [oldW+newW for oldW, newW in zip(self.neuralNet.weightsLayers, sumWeightsDeriv)]

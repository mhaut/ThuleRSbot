# import 
import src.Algorithms._Base.BaseClassifiers as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg
from src.Algorithms.Classification.Supervised.NeuralNetworks import BaseNeuralNetworks as BN
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.layers import Dense


class MultiLayerPerceptron(BN.NeuralNetworks):

    def get_model_compiled(self, n_bands, num_class, opt):
        clf = Sequential()
        clf.add(Dense(int(n_bands[0] * 2 / 3.) + 10, activation='relu', input_shape=(n_bands[0],)))
        clf.add(Dense(num_class, activation='softmax'))
        clf.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
        return clf


class MLP(BC.BaseClassifiers):
    '''
    Class of algorithm Multilayer Perceptron
    '''

    def __init__(self, device='CPU', optimizer='Adam', epochs=200, batch_size=100):
        
        self.device = device
        self.typeAlgorithm = enumTypeAlg.SUPERVISED
        self.model = MultiLayerPerceptron(device=device, optimizer=optimizer,
                                       epochs=epochs, batch_size=batch_size)

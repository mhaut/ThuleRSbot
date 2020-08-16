# import 
import src.Algorithms._Base.BaseClassifiers as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg
from src.Algorithms.Classification.Supervised.NeuralNetworks import BaseNeuralNetworks as BN
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, Conv2D, MaxPooling2D
from keras import regularizers

        
class ConvolutionalNeuralNetwork2D(BN.NeuralNetworks):

    def get_model_compiled(self, n_bands, num_class, opt):
        clf = Sequential()
        clf.add(Conv2D(50, kernel_size=(5, 5), input_shape=n_bands))
        clf.add(Activation('relu'))
        clf.add(Conv2D(100, (5, 5)))
        clf.add(Activation('relu'))
        clf.add(MaxPooling2D(pool_size=(2, 2)))
        clf.add(Flatten())
        clf.add(Dense(100, kernel_regularizer=regularizers.l2()))
        clf.add(Activation('relu'))
        clf.add(Dense(num_class, activation='softmax'))
        clf.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
        return clf
    
    def preProcesingDataFit(self, x_train, y_train):
        #TODO
        return x_train, y_train
    
    def preProcesingDataPredit(self, x_test):
        #TODO
        return x_test
    
    def preProcesingDataScore(self, x_test, y_test):
        #TODO
        return x_test, y_test


class CNN2D(BC.BaseClassifiers):
    '''
    Class of CNN1
    '''

    def __init__(self, device='CPU', optimizer='Adam', epochs=200, batch_size=100):
        
        self.device = device
        self.typeAlgorithm = enumTypeAlg.SUPERVISED
        self.model = ConvolutionalNeuralNetwork2D(device=device, optimizer=optimizer,
                                       epochs=epochs, batch_size=batch_size)

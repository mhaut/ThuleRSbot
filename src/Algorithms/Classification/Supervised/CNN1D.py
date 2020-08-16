# import 
import src.Algorithms._Base.BaseClassifiers as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg
from src.Algorithms.Classification.Supervised.NeuralNetworks import BaseNeuralNetworks as BN
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Conv1D, Dense, Flatten, MaxPooling1D
    
    
class ConvolutionalNeuralNetwork1D(BN.NeuralNetworks):

    def get_model_compiled(self, n_bands, num_class, opt):
        clf = Sequential()
        clf.add(Conv1D(20, (24), activation='relu', input_shape=(n_bands[0], 1)))
        clf.add(MaxPooling1D(pool_size=5))
        clf.add(Flatten())
        clf.add(Dense(100))
        clf.add(BatchNormalization())
        clf.add(Activation('relu'))
        clf.add(Dense(num_class, activation='softmax'))
        clf.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
        return clf

    def preProcesingDataFit(self, x_train, y_train):
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        return x_train, y_train
    
    def preProcesingDataPredit(self, x_test):
        return x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    
    def preProcesingDataScore(self, x_test, y_test):
        return x_test.reshape(x_test.shape[0], x_test.shape[1], 1), y_test


class CNN1D(BC.BaseClassifiers):
    '''
    Class of CNN1
    '''

    def __init__(self, device='CPU', optimizer='Adam', epochs=200, batch_size=100):
        
        self.device = device
        self.typeAlgorithm = enumTypeAlg.SUPERVISED
        self.model = ConvolutionalNeuralNetwork1D(device=device, optimizer=optimizer,
                                       epochs=epochs, batch_size=batch_size)

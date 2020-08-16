# import 
import src.Algorithms._Base.BaseClassifiers as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg
from src.Algorithms.Classification.Supervised.NeuralNetworks import BaseNeuralNetworks as BN
from keras.applications import resnet50, vgg16, DenseNet121
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.layers import Dense
import keras

class CNN_pretrained(BN.NeuralNetworks):

    def get_model_compiled(self, shape, num_class, opt): #AVISO shape basura
        # build model
        arch='resnet18' #AVISO
        if "resnet" in arch:
            if arch == "resnet18":
                base_model = resnet50.ResNet50(input_shape=(224,224,3), weights='imagenet', include_top=False)
            elif arch == "resnet50":
                base_model = resnet18.ResNet50(input_shape=(224,224,3), weights='imagenet', include_top=False)
            else:
                print("ERROR modelo resnet no existe")
            x = keras.layers.GlobalAveragePooling2D()(base_model.output)
            output = keras.layers.Dense(num_class, activation='softmax')(x)
            model = keras.models.Model(inputs=[base_model.input], outputs=[output])
        else:
            print("ERROR modelo CNN no existe")
        model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy'])
        return model


class CNN_RGB(BC.BaseClassifiers):
    '''
    Class of algorithm Multilayer Perceptron
    '''

    def __init__(self, device='CPU', arch='resnet18', optimizer='SGD', epochs=200, batch_size=100):
        
        self.device = device
        self.typeAlgorithm = enumTypeAlg.SUPERVISED
        self.model = CNN_pretrained(device=device, optimizer=optimizer,
                                       epochs=epochs, batch_size=batch_size)
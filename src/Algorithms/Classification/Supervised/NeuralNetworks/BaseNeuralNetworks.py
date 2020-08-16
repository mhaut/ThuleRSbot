# import 
import os
from uuid import uuid4
from datetime import datetime
from src.Algorithms.Classification.Supervised.NeuralNetworks import Optimizers

import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import to_categorical as keras_to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import gc


def select_device_keras(device):
    if device == "CPU":
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device.startswith('GPU'):
        os.environ["CUDA_VISIBLE_DEVICES"] = device[3]
    else:
        print("Device not found")
        return


class GeneralNeuralNetworks():

    def __init__(self, device, optimizer=None, lr=None, decay=None, momentum=None,
                 nesterov=None, rho=None, beta_1=None, beta_2=None, amsgrad=None,
                 epochs=200, batch_size=100):
        self.epochs = epochs
        self.batch_size = batch_size
        self.idProcess = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
        select_device_keras(device)
        # None is Default parameters
        self.classOptimizer = Optimizers.myoptimizer(method=optimizer, lr=lr, decay=decay, momentum=momentum,
                              nesterov=nesterov, rho=rho, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad)

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        num_class = len(np.unique(y_train))
        x_train, y_train = self.preProcesingDataFit(x_train, y_train)
        self.model = self.get_model_compiled(x_train.shape[1:], num_class, self.classOptimizer.opt_select)
        
        # TODO: sacar el history (esto lo  hago yo)
        if x_val != None:
            path_save_model = "/tmp/best_model" + str(self.idProcess) + ".h5"
            history = self.model.fit(x_train, keras_to_categorical(y_train, num_class),
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=0,
                            validation_data=(x_val, keras_to_categorical(y_val, num_class)),
                            callbacks=[ModelCheckpoint(path_save_model, monitor='val_accuracy', verbose=0, save_best_only=True)])
            self.model = load_model(path_save_model)
            os.remove(path_save_model)
        else:
            history = self.model.fit(x_train, keras_to_categorical(y_train, num_class),
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=0)

    def predict(self, data):
        data = self.preProcesingDataPredit(data)
        return np.argmax(self.model.predict(data, verbose=0), axis=1)
    
    def score(self, data, labels):
        data, labels = self.preProcesingDataScore(data, labels)
        score, acc = self.model.evaluate(data, keras_to_categorical(labels, len(np.unique(labels))), verbose=0)
        return acc
        
    def get_model_compiled(self, n_bands, num_class, opt):
        pass
    
    def preProcesingDataFit(self, x_train, y_train):
        return x_train, y_train
    
    def preProcesingDataPredit(self, x_test):
        return x_test
    
    def preProcesingDataScore(self, x_test, y_test):
        return x_test, y_test
    
    def __del__(self):
        try:
            del self.model
            K.clear_session()
            gc.collect()
        except:
            pass


class NeuralNetworks(GeneralNeuralNetworks):

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        num_class = len(np.unique(y_train))
        x_train, y_train = self.preProcesingDataFit(x_train, y_train)
        self.model = self.get_model_compiled(x_train.shape[1:], num_class, self.classOptimizer.opt_select)
        
        # TODO: sacar el history (esto lo  hago yo)
        if x_val != None:
            path_save_model = "/tmp/best_model" + str(self.idProcess) + ".h5"
            history = self.model.fit(x_train, keras_to_categorical(y_train, num_class),
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=0,
                            validation_data=(x_val, keras_to_categorical(y_val, num_class)),
                            callbacks=[ModelCheckpoint(path_save_model, monitor='val_accuracy', verbose=0, save_best_only=True)])
            self.model = load_model(path_save_model)
            os.remove(path_save_model)
        else:
            history = self.model.fit(x_train, keras_to_categorical(y_train, num_class),
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=0)


class NeuralNetworksPretrained(GeneralNeuralNetworks):

    def __init__(self, parent=None):
        super(GeneralNeuralNetworks, self).__init__(parent)
        self.datagen_tr = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
        self.datagen_te = ImageDataGenerator()

    def random_crop(self, img, random_crop_size):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y + dy), x:(x + dx), :]

    def center_crop(self, img, random_crop_size):        
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        y = int((height - random_crop_size[0]) / 2)
        x = int((width - random_crop_size[1]) / 2)
        return img[x:x + random_crop_size[0], y:y + random_crop_size[1], :]

    def crop_generator(self, batches, crop_length, func="rand"):
        """Take as input a Keras ImageGen (Iterator) and generate random
        crops from the image batches generated by the original iterator.
        """
        while True:
            batch_x, batch_y = next(batches)
            batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
            for i in range(batch_x.shape[0]):
                if func == "rand":
                    batch_crops[i] = self.random_crop(batch_x[i], (crop_length, crop_length))
                else:
                    batch_crops[i] = self.center_crop(batch_x[i], (crop_length, crop_length))
            yield (batch_crops, batch_y)

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        num_class = len(np.unique(y_train))
        x_train, y_train = self.preProcesingDataFit(x_train, y_train)
        
        train_batches = self.datagen_tr.flow(x_train, y_train, shuffle=True, batch_size=self.batch_size,)
        test_batches = self.datagen_te.flow(x_val, y_val, shuffle=False, batch_size=self.batch_size,)
        train_crops = self.crop_generator(train_batches, 224, func="rand")
        test_crops = self.crop_generator(test_batches, 224, func="center")
        
        self.model = self.get_model_compiled(num_class, self.classOptimizer.opt_select)
        
        # TODO: sacar el history (esto lo  hago yo)
        if x_val != None:
            path_save_model = "/tmp/best_model" + str(self.idProcess) + ".h5"
            history = self.model.fit(train_crops,
                    steps_per_epoch=np.ceil(len(x_train) / self.batch_size), epochs=self.epochs,
                    verbose=0,
                    validation_data=test_crops,
                    validation_steps=np.ceil(len(x_val) / self.batch_size),
                    callbacks=[ModelCheckpoint(path_save_model, monitor='val_accuracy', verbose=0, save_best_only=True)])
            self.model = load_model(path_save_model)
            os.remove(path_save_model)
        else:
            self.model.fit(train_crops,
                    steps_per_epoch=np.ceil(len(x_train) / self.batch_size), epochs=self.epochs,
                    verbose=0,
                    validation_data=test_crops,
                    validation_steps=np.ceil(len(x_val) / self.batch_size))

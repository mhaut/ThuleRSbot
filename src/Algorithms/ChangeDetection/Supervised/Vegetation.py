# import 
import src.Algorithms._Base.BaseChangeDetection as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg
from colormap import Colormap
import numpy as np


class LibVegetation():

    def __init__(self):
        self.c = Colormap()
        # Vegetation
        self.colorMapDatas = self.c.cmap_linear('Beige', 'red', 'green')
        self.colorMapResult = self.c.cmap_linear('Beige', 'green', 'red')
        self.colorMapRGB = 'jet'

    def execution(self, data1, data2, labels1=[], labels2=[]):
        if len(labels1) != 0 and len(labels2) != 0:
            labels1[labels1 != 4] = 0
            labels2[labels2 != 4] = 0
            vegetationTotal = np.abs(labels1 - labels2)
            return labels1, labels2, vegetationTotal
        else:
            raise Exception("No exist the labels")


class Vegetation(BC.BaseChangeDetection):
    '''
    Class of algorithm Vegetation
    '''

    def __init__(self, device='CPU'):

        self.model = LibVegetation()
        self.device = device
        self.typeAlgorithm = enumTypeAlg.SUPERVISED

# import 
import src.Algorithms._Base.BaseChangeDetection as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg
import numpy as np

class LibChlorophyll():
    def __init__(self):
        self.colorMapDatas = self.colorMapResult = self.colorMapRGB = 'jet'

    def execution(self, data1, data2, labels1=[], labels2=[]):
        # chlorophyll
        chlorophyll1 = (data1[:, :, 8] - data1[:, :, 3]) / (data1[:, :, 8] + data1[:, :, 3])
        chlorophyll2 = (data2[:, :, 8] - data2[:, :, 3]) / (data2[:, :, 8] + data2[:, :, 3])
        chlorophyllTotal = np.abs(chlorophyll2 - chlorophyll1)
        chlorophyllTotal[chlorophyllTotal < 0.05] = 0  # filtrar ligeras variaciones
        return chlorophyll1, chlorophyll2, chlorophyllTotal


class Chlorophyll(BC.BaseChangeDetection):
    '''
    Class of algorithm Vegetation
    '''

    def __init__(self, device='CPU'):

        self.model = LibChlorophyll()
        self.device = device
        self.typeAlgorithm = enumTypeAlg.UNSUPERVISED

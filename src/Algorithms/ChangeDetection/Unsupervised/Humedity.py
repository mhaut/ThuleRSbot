# import 
import src.Algorithms._Base.BaseChangeDetection as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg
import numpy as np

class LibHumedity():
    def __init__(self):
        self.colorMapDatas = self.colorMapResult = self.colorMapRGB = 'jet'

    def execution(self, data1, data2, labels1=[], labels2=[]):
        # Humedity
        Humedity1 = (data1[:, :, 8] - data1[:, :, 11]) / (data1[:, :, 8] + data1[:, :, 11])
        Humedity2 = (data2[:, :, 8] - data2[:, :, 11]) / (data2[:, :, 8] + data2[:, :, 11])
        HumedityTotal = np.abs(Humedity2 - Humedity1)
        HumedityTotal[HumedityTotal < 0.05] = 0  # filtrar ligeras variaciones
        return Humedity1, Humedity2, HumedityTotal


class Humedity(BC.BaseChangeDetection):
    '''
    Class of algorithm Vegetation
    '''

    def __init__(self, device='CPU'):

        self.model = LibHumedity()
        self.device = device
        self.typeAlgorithm = enumTypeAlg.UNSUPERVISED
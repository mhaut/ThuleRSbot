import numpy as np
from scipy.optimize import lsq_linear
import src.Algorithms._Base.BaseUnmixing as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg


class LibLSU():

    def __init__(self, delta=1 / 10000.):
        self.components_ = None
        self.delta = delta

    def fit(self, _ , components):
        self.components_ = components

    def transform(self, data):
        npixel, _ = data.shape
        nbands, nendmembers = self.components_.shape

        N = np.zeros((nbands + 1, nendmembers), float)
        N[:nbands, :nendmembers] = self.delta * self.components_
        N[nbands, :] = np.ones((nendmembers), float)
        s = np.zeros((nbands + 1, 1), float)
        abundances = np.zeros((npixel, nendmembers), float) 
        for p in range(0, npixel):
                s[:nbands, 0] = self.delta * np.squeeze(data[p, :])
                s[nbands] = 1
                result = lsq_linear(N, s[:, 0])
                abundances[p, :] = result['x']  # obtenemos el primer parametro retornado
        return abundances


class LSU(BC.BaseUnmixing):
    '''
    Class of algorithm LSU
    '''

    def __init__(self, device='CPU', delta=1 / 10000.):
        self.model = LibLSU(delta=delta)
        self.device = device
        self.typeAlgorithm = enumTypeAlg.SUPERVISED

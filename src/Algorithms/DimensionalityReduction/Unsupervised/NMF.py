# import 
from sklearn.decomposition import NMF as LibNMF
import src.Algorithms._Base.BaseDimensionalityReduction as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg


class NMF(BC.BaseDimensionalityReduction):
    '''
    Class of algorithm NMF
    '''

    def __init__(self, device='CPU', n_components=None, init=None, solver='cd', beta_loss='frobenius', tol=0.0001, max_iter=200,
                random_state=None, alpha=0.0, l1_ratio=0.0, verbose=0, shuffle=False):
        self.model = LibNMF(n_components, init, solver, beta_loss, tol, max_iter, random_state, alpha, l1_ratio, verbose, shuffle)
        self.device = device
        self.typeAlgorithm = enumTypeAlg.UNSUPERVISED

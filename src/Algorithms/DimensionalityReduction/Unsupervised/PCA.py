# import 
from sklearn.decomposition import PCA as LibPCA
import src.Algorithms._Base.BaseDimensionalityReduction as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg


class PCA(BC.BaseDimensionalityReduction):
    '''
    Class of algorithm PCA
    '''

    def __init__(self, device='CPU', n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto',
                random_state=None):
        self.model = LibPCA(n_components, copy, whiten, svd_solver, tol, iterated_power, random_state)
        self.device = device
        self.typeAlgorithm = enumTypeAlg.UNSUPERVISED

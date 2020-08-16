# import 
from sklearn.decomposition import FastICA as LibFastICA
import src.Algorithms._Base.BaseDimensionalityReduction as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg


class FastICA(BC.BaseDimensionalityReduction):
    '''
    Class of algorithm FastICA
    '''

    def __init__(self, device='CPU', n_components=None, algorithm='parallel', whiten=True, fun='logcosh', fun_args=None,
                max_iter=200, tol=0.0001, w_init=None, random_state=None):
        self.model = LibFastICA(n_components, algorithm, whiten, fun, fun_args, max_iter, tol, w_init, random_state)
        self.device = device
        self.typeAlgorithm = enumTypeAlg.UNSUPERVISED

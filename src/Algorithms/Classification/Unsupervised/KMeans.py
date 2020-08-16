# import 
from sklearn.cluster import KMeans as LibKMeans
import src.Algorithms._Base.BaseClassifiers as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg


class KMeans(BC.BaseClassifiers):
    '''
    Class of algorithm KMeans
    '''

    def __init__(self, device='CPU', n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto'):
        self.model = LibKMeans(n_clusters, init, n_init, max_iter, tol, precompute_distances, verbose, random_state, copy_x,
                    n_jobs, algorithm)
        self.device = device
        self.typeAlgorithm = enumTypeAlg.UNSUPERVISED

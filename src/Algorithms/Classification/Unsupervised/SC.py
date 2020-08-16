# import 
from sklearn.cluster import SpectralClustering
import src.Algorithms._Base.BaseClassifiers as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg


class SC(BC.BaseClassifiers):
    '''
    Class of algorithm Spectral Clustering
    '''

    def __init__(self, device='CPU', n_clusters=8, eigen_solver=None, random_state=None, n_init=1, gamma=1.0,
                 affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1,
                 kernel_params=None, n_jobs=None):
        self.model = SpectralClustering(n_clusters, eigen_solver, random_state, n_init, gamma, affinity, n_neighbors,
                                eigen_tol, assign_labels, degree, coef0, kernel_params, n_jobs)
        self.device = device
        self.typeAlgorithm = enumTypeAlg.UNSUPERVISED

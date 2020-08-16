# import 
from sklearn.neighbors import KNeighborsClassifier
import src.Algorithms._Base.BaseClassifiers as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg


class KNN(BC.BaseClassifiers):
    '''
    Class of algorithm k-nearest neighbors
    '''

    def __init__(self, device='CPU', n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30,
                p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs):

        self.model = KNeighborsClassifier(n_neighbors, weights, algorithm, leaf_size,
                p, metric, metric_params, n_jobs, **kwargs)
        self.device = device
        self.typeAlgorithm = enumTypeAlg.SUPERVISED

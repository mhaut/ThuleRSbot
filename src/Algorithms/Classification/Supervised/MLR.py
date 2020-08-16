# import 
from sklearn.linear_model import LogisticRegression
import src.Algorithms._Base.BaseClassifiers as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg


class MLR(BC.BaseClassifiers):
    '''
    Class of algorithm Multiple Linear Regression
    '''

    def __init__(self, device='CPU', penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                class_weight=None, random_state=None, solver='warn', max_iter=100, multi_class='warn', verbose=0,
                warm_start=False, n_jobs=None, l1_ratio=None):
        self.model = LogisticRegression(penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight,
                                random_state, solver, max_iter, multi_class, verbose, warm_start, n_jobs, l1_ratio)
        self.device = device
        self.typeAlgorithm = enumTypeAlg.SUPERVISED

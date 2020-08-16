# import 
from sklearn.ensemble import RandomForestClassifier
import src.Algorithms._Base.BaseClassifiers as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg


class RF(BC.BaseClassifiers):
    '''
    Class of algorithm Random Forest Classifier
    '''

    def __init__(self, device='CPU', n_estimators='warn', criterion='gini', max_depth=None, min_samples_split=2,
                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                warm_start=False, class_weight=None):
        self.model = RandomForestClassifier(n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf,
                                    min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease,
                                    min_impurity_split, bootstrap, oob_score, n_jobs, random_state, verbose,
                                    warm_start, class_weight)
        self.device = device
        self.typeAlgorithm = enumTypeAlg.SUPERVISED

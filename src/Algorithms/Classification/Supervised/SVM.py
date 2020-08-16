# import 
import src.Algorithms._Base.BaseClassifiers as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg


class SVM(BC.BaseClassifiers):
    '''
    Class of algorithm Support Vector Machines 
    '''
    
    def __init__(self, device='CPU', C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, shrinking=True,
                probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
                max_iter=-1, decision_function_shape='ovr', random_state=None):
        
        kwargs = {'gamma':gamma, 'C':C, 'coef0':coef0, 'tol':tol, 'max_iter':max_iter, 'kernel':kernel,
                    'degree':degree, 'shrinking':shrinking, 'cache_size':cache_size, 'class_weight':class_weight, \
                    'probability':probability, 'decision_function_shape':decision_function_shape, \
                    'random_state':random_state, 'verbose':verbose}
        self.device = device
        self.typeAlgorithm = enumTypeAlg.SUPERVISED
        
        if self.device.startswith('GPU'): 
            from thundersvm import SVC
        else: 
            from sklearn.svm import SVC
        
        self.model = SVC(**kwargs)

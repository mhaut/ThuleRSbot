#import
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg
import src.Algorithms._Base.'#PutBase' as BC
'#Put your imports'

class YourAlgorithms(BC.'#PutBase'):
    '''
    Class of algorithm YourAlgorithms
    '''
    def __init__(self, device='CPU or GPU', param1=defaultParam1, param2=defaultParam2,...,paramN=defaultParamN):
        self.model = PutClassThatImplementsAlgorithm(param1, param2, ..., paramN)
        self.device = device
        self.typeAlgorithm = enumTypeAlg.'#PutType'
        
        
'''
Replace #PutBase with one of the following options
    BaseClassifiers: for Classification algorithms
    BaseDimensionalityReduction: for Dimensionality Reduction algorithms
    BaseRestoration: for Restoration algorithms
    BaseUnmixing: for Unmixing algorithms
    BaseChangeDetection: for Change Detection algorithms
        
Replace #PutType with one of the following options
    SUPERVISED: for supervised algorithms
    UNSUPERVISED: for unsupervised algorithms
    SEMISUPERVISED: for semisupervised algorithms
'''

'''
PutClassThatImplementsAlgorithm can be a class with your algorithm implementation and would be put in this file or it can be an algorithm that is implemented in some python library (import to that library is necessary)
'''


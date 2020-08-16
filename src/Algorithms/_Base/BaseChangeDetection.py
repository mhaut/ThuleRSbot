import time
from numba import cuda as numba_cuda


class BaseChangeDetection():

    def __init__(self):
        '''
        Builder
        Parameters:
            Input: 
                model -> algorithm of machine learning
                typeAlgorithm -> 0 if algorithm is Supervised; 1 if algorithm is Unsupervised; 2 if algorithm is semisupervised    (enumerate class in Algorithms.__init__.py
                device -> CPU or GPU       
        '''
        self.model = None
        self.typeAlgorithm = None
        self.device = 'CPU'

    def execution(self, data1, data2, labels1=[], labels2=[]):
        execution_time1 = time.time()
        if len(labels1) == 0 and len(labels2) == 0: 
            result = self.model.execution(data1, data2)
        else: 
            result = self.model.execution(data1, data2, labels1, labels2)
        execution_time = time.time() - execution_time1
        return result, execution_time
    
    def __del__(self):
        try:
            del self.model, self.typeAlgorithm
            if self.device == "GPU":
                numba_cuda.select_device(0)
                numba_cuda.close()
            del self.device
        except:
            pass

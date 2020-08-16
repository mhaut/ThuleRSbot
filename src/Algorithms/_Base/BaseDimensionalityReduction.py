import time
from numba import cuda as numba_cuda

class BaseDimensionalityReduction():

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
    
    def train_algorithm(self, data, labels=[]):
        '''
        Train algorithm
        Parameters: 
            Input: 
                data -> matrix: shape(pixel, bands)
                labels -> array: shape(pixel) [Necessary in typeAlgorithm]
            Output: 
                train_time -> time of execution
        '''
        train_time = time.time()
        if labels == []:
            self.model.fit(data)
        else: 
            self.model.fit(data, labels)
        train_time = time.time() - train_time
        return train_time

    def test_algorithm(self, data):
        '''
        Test algorithm
        Parameters: 
            Input: 
                data -> matrix: shape(pixel, bands)
            Output: 
                predicted -> result of algorithm: matrix: shape(pixel, bands) 
                test_time -> time of execution
        '''
        test_time = time.time()
        predicted = self.model.transform(data)
        test_time = time.time() - test_time
        return predicted, test_time

    def score(self, data, labels=[]):
        '''
        Score of algorithm
            Input: 
                data -> matrix: shape(pixel, bands)
                labels -> array: shape(pixel) [Necessary in typeAlgorithm]
            Output: 
                score -> algorithm accuracy
        '''
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(data, self.model.inverse_transform(self.model.transform(data)))
    
    def __del__(self):
        try:
            del self.model, self.typeAlgorithm
            if self.device == "GPU":
                numba_cuda.select_device(0)
                numba_cuda.close()
            del self.device
        except:
            pass
    

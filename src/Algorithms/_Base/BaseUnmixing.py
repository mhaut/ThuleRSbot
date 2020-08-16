import time
import copy
import numpy as np
import scipy.io
from sklearn.metrics import mean_squared_error
from sklearn.utils.extmath import softmax
import scipy.spatial.distance as sp_dist
from numpy.linalg import norm
from numba import cuda as numba_cuda


class BaseUnmixing():

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

    def test_algorithm(self, data, signatureEndmembers=[]):
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
        if signatureEndmembers == []:
            predicted = self.model.transform(data)
        else:
            predicted = self.model.transform(data, signatureEndmembers)
        test_time = time.time() - test_time
        return predicted, test_time

    def score(self, data, labels=None):
        '''
        Score of algorithm
            Input: 
                data -> matrix: shape(pixel, bands)
                labels -> array: shape(pixel) [Necessary in typeAlgorithm]
            Output: 
                score -> algorithm accuracy (rmse)
        '''
        rmse = None
        if ('A' in scipy.io.loadmat(labels).keys()):
            abundancesGT = scipy.io.loadmat(labels)['A'].T
            # materiales
            K = abundancesGT.shape[1]
        
            if(not self.typeAlgorithm):
                    # Pair predicted/true equal abundances before RMSE
                    image_s1 = data.reshape(-1, K)
                    image_gt = abundancesGT.reshape(-1, K)
                    dists = []
        
                    for col in range(image_s1.shape[1]):
                        act_sim = []
                        row = image_s1[:, col]
                        for col2 in range(image_gt.shape[1]):
                            row2 = image_gt[:, col2]
                            act_sim.append(sp_dist.cosine(row, row2))
                        dists.append(act_sim)
                    dists = np.array(dists)
                    new_classes = [0] * K
                    ab2 = copy.deepcopy(data)
        
                    for _ in range(K):
                        (fil, col) = np.unravel_index(dists.argmin(), dists.shape)
                        data[:, col] = ab2[:, fil]
                        new_classes[fil] = col
                        dists[:, col] = 100000
                        dists[fil, :] = 100000
                    del ab2, new_classes, dists, image_gt, image_s1
                
            rmse = np.sqrt(mean_squared_error(abundancesGT.reshape(-1, K), data.reshape(-1, K)))
            
        return rmse

    def scoreSad(self, endmembersPredicted, dataset, path_abundances_GT=None):
        '''
        Another Score of algorithm (only for unsupervised algorithms)
            Input:
                endmembersPredicted -> matrix: shape(bands, endmembers)
                dataset -> path of dataset
                path_abundances_GT -> path of abundances_GT
            Output: 
                score -> algorithm accuracy (sad)
        '''
    
        if(not self.typeAlgorithm):
            
            K = endmembersPredicted.shape[1]
            endmembersGT = scipy.io.loadmat(path_abundances_GT)['M']
        
            if dataset == 'PublicDatasets/Unmixing/CupriteS1_R188.mat':
                bands = scipy.io.loadmat(path_abundances_GT)['slctBnds'][0, :]
                endmembersGT = endmembersGT[bands]
                softmaxed = softmax(endmembersGT.T)
                endmembersGT = softmaxed.T
    
            if (path_abundances_GT != None):
                # Pair predicted/true equal endmembers before SAD
                endm_s1 = endmembersPredicted
                endm_gt = endmembersGT
        
                dists = []
        
                for col in range(endm_s1.shape[1]):
                    act_sim = []
                    row = endm_s1[:, col]
                    for col2 in range(endm_gt.shape[1]):
                        row2 = endm_gt[:, col2]
                        act_sim.append(sp_dist.cosine(row, row2))
                    dists.append(act_sim)
        
                dists = np.array(dists)
                new_classes = [0] * K
                en2 = copy.deepcopy(endmembersPredicted)
        
                for i in range(K):
                    (fil, col) = np.unravel_index(dists.argmin(), dists.shape)
                    endmembersPredicted[:, col] = en2[:, fil]
                    new_classes[fil] = col
                    dists[:, col] = 100000
                    dists[fil, :] = 100000
        
                del en2, new_classes, dists, endm_gt, endm_s1
                
            cos_sim = 0
            
            for i in range(K):
                b = endmembersGT[:, i]
                a = endmembersPredicted[:, i]
                cos_sim += np.arccos(np.dot(a, b) / (norm(a) * norm(b)))
            sad = cos_sim / float(K)
        else:
            sad = None
            
        return sad
    
    def __del__(self):
        try:
            del self.model, self.typeAlgorithm
            if self.device == "GPU":
                numba_cuda.select_device(0)
                numba_cuda.close()
            del self.device
        except:
            pass

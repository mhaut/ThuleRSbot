# import 
import numpy as np
import time
import src.Algorithms._Base.BaseUnmixing as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg


class LibPLSA():

    def __init__(self, n_components=None, max_iter=2000,
                    reg1=0.0, reg2=0.0, random_state=None, devicePLSA="CPU"):

        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        # lamda[i, j] : p(zj|di)
        self.lamda = None
        # theta[i, j] : p(wj|zi)
        self.theta = None
        # ALERT:PARA CUADRAR LA LIBRERIA CON LDA
        self.components_ = self.theta
        # p[i, j, k] : p(zk|di,wj)
        self.p = None
        self.reg1 = reg1
        self.reg2 = reg2
        self.devicePLSA = devicePLSA

    def EStep(self, N, M, p, n_components):
        for i in range(0, N):  # por cada pixel
            for j in range(0, M):  # por cada banda
                denominator = 0;
                for k in range(0, n_components):  # por cada endmember
                    p[i, j, k] = self.theta[k, j] * self.lamda[i, k];
                    denominator += p[i, j, k];
                if denominator == 0:
                    for k in range(0, n_components):
                        p[i, j, k] = 0;
                else:
                    for k in range(0, n_components):
                        p[i, j, k] /= denominator
        return p

    def MStep(self, X, N, M, p, n_components, regularization1, regularization2):
        # update self.theta
        for k in range(0, n_components):  # por cada endmember
            denominator = 0
            for j in range(0, M):  # de cada banda
                self.theta[k, j] = 0
                for i in range(0, N):  # calcular self.thetas para cada pixel
                    valueaux = X[i, j] * p[i, j, k] - (regularization1 / float(M))
                    if valueaux > 0:
                        self.theta[k, j] += valueaux
                denominator += self.theta[k, j]
            if denominator == 0:
                for j in range(0, M):
                    self.theta[k, j] = 1.0 / M
            else:
                for j in range(0, M):
                    self.theta[k, j] /= denominator
        # update self.lamda
        for i in range(0, N):
            for k in range(0, n_components):
                self.lamda[i, k] = 0
                denominator = 0
                for j in range(0, M):
                    valueaux = X[i, j] * p[i, j, k] - (regularization2 / float(n_components))
                    if valueaux > 0:
                        self.lamda[i, k] += valueaux
                    denominator += X[i, j];
                if denominator == 0:
                    self.lamda[i, k] = 1.0 / n_components
                else:
                    self.lamda[i, k] /= denominator
        return self.theta, self.lamda

    def fit(self, data):
        if self.devicePLSA.startswith('GPU'):
            # #Traceback (most recent call last):
            # #File "/home/greta/.local/lib/python3.6/site-packages/pycuda/autoinit.py", line 14, in _finish_up
                # #context.pop()
            # #pycuda._driver.LogicError: context::pop failed: invalid device context - cannot pop non-current context
            ##-------------------------------------------------------------------
            # #PyCUDA ERROR: The context stack was not empty upon module cleanup.
            ##-------------------------------------------------------------------
            # #A context was still active when the context stack was being
            # #cleaned up. At this point in our execution, CUDA may already
            # #have been deinitialized, so there is no way we can finish
            # #cleanly. The program will be aborted now.
            # #Use Context.pop() to avoid this problem.
            ##-------------------------------------------------------------------
            #import sys
            #sys.path.append("Algorithms/Unmixing/Unsupervised/plsagpu/")
            import src.Algorithms.Unmixing.Unsupervised.plsagpu.plsa_cuda as plsa_cuda
            # import pycuda.driver as cuda
            (p, denominators, theta, lamda, N, M) = plsa_cuda.generate_inputs(data, self.n_components)
            # cxt = plsa_cuda.initGPU()
            self.components_, self.ab = plsa_cuda.pLSACUDA(lamda, theta, p, data.astype("float32"), denominators, N, M, self.n_components, self.max_iter, self.reg1, self.reg2)
            # cxt.pop()
        else:
            N, M = data.shape
            np.random.RandomState(seed=self.random_state)
            self.lamda = np.random.random((N, self.n_components))  # abundances
            self.theta = np.random.random((self.n_components, M))  # endmembers
            self.p = np.zeros((N, M, self.n_components))  # posterior

            # normalizacion de parametros lambda y self.theta
            for i in range(0, N):
                normalization = sum(self.lamda[i, :])
                for j in range(0, self.n_components):
                    self.lamda[i, j] /= normalization
            for i in range(0, self.n_components):
                normalization = sum(self.theta[i, :])
                for j in range(0, M):
                    self.theta[i, j] /= normalization

            oldLoglikelihood = 1; newLoglikelihood = 1
            for i in range(self.max_iter):
                self.p = self.EStep(N, M, self.p, self.n_components)
                self.theta, self.lamda = self.MStep(data, N, M, self.p, self.n_components, self.reg1, self.reg2)
            # ALERT:PARA CUADRAR LA LIBRERIA CON LDA
            self.components_ = self.theta
        
    def transform(self, data):
        if self.devicePLSA.startswith('GPU'): return self.ab
        else: return np.dot(data.astype("float32"), self.components_.T)


class PLSA(BC.BaseUnmixing):
    '''
    Class of algorithm PLSA
    '''

    def __init__(self, n_endmembers=None, device='CPU', max_iter=2000, reg1=0.0, reg2=0.0, random_state=None):
        self.device = device
        self.typeAlgorithm = enumTypeAlg.UNSUPERVISED
        self.model = LibPLSA(n_components=n_endmembers, max_iter=max_iter, reg1=reg1, reg2=reg2, random_state=random_state, devicePLSA=self.device)

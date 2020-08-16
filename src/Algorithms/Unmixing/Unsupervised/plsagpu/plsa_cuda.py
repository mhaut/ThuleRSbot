# Importing PLSA
# import time
import numpy as np
from math import ceil
import pycuda.autoinit
from pycuda.gpuarray import to_gpu
from kernels import kernels
import pycuda.driver as cuda
from numba import cuda as numba_cuda
########################################################################################
##########################     Functions      #########################################
########################################################################################


def initPycuda():
    # Initialize CUDA
    cuda.init()
    from pycuda.tools import make_default_context
    context = make_default_context()
    device = context.get_device()

    import atexit
    atexit.register(context.pop)
    return context, device, atexit


def generate_inputs(image, K):
    N, M = image.shape
    # lamda[i, j] : p(zj|di)
    lamda = np.random.uniform(size=(N, K))  # abundances
    # theta[i, j] : p(wj|zi)
    theta = np.random.random([K, M])  # endmembers
    # p[i, j, k] : p(zk|di,wj)
    p = np.zeros([N, M, K])  # posterior
    denominators = np.zeros(K)
    
    # normalizacion de parametros lambda y theta
    for i in range(0, N):
        normalization = sum(lamda[i, :])
        for j in range(0, K):
            lamda[i, j] /= normalization
    for i in range(0, K):
        normalization = sum(theta[i, :])
        for j in range(0, M):
            theta[i, j] /= normalization
    lamda = lamda.astype(np.float32)
    theta = theta.astype(np.float32)
    p = p.astype(np.float32)
    
    return p, denominators, theta, lamda, N, M


def pLSACUDA(lamda, theta, p, X, denominators, N, M, K, iters, r1, r2, cxt=None):
    lamda_gpu = to_gpu(lamda)
    theta_gpu = to_gpu(theta)
    p_gpu = to_gpu(p)
    X_gpu = to_gpu(X)
    den_gpu = to_gpu(denominators)
    steps = int(ceil(N / 1024.0))
    
    EStep = kernels.get_function("EStep")
    LamdaComputing = kernels.get_function("Lamda")
    ThetaComputing = kernels.get_function("Theta1")
    CalculeDiv = kernels.get_function("Theta2")
    ThetaDivision = kernels.get_function("Theta3")
    
    # if cxt == None: pycuda.driver.Context.synchronize()
    # else: cxt.synchronize()
    # context, device, atexit = initPycuda()
    
    for i in range(iters):
        EStep(p_gpu, theta_gpu, lamda_gpu, block=(1, 1, K), grid=(N, M, 1))
        ThetaComputing(theta_gpu, X_gpu, p_gpu, np.float32(r1), np.uint32(steps), np.uint32(N), np.uint32(M), den_gpu, block=(1024, 1, 1), grid=(1, M, K))
        CalculeDiv(theta_gpu, den_gpu, block=(1, M, 1), grid=(1, 1, K))
        ThetaDivision(theta_gpu, den_gpu, block=(1, 1, 1), grid=(1, M, K))
        LamdaComputing(lamda_gpu, X_gpu, p_gpu, np.uint32(K), np.float32(r2), block=(1, M, 1), grid=(N, 1, K))
    
    # if cxt == None: pycuda.driver.Context.synchronize()
    # else: cxt.synchronize()

    abundances = lamda_gpu.get().reshape(X.shape[0], K)
    endmembers = theta_gpu.get()

    lamda_gpu.gpudata.free()
    theta_gpu.gpudata.free()
    X_gpu.gpudata.free()
    p_gpu.gpudata.free()
    den_gpu.gpudata.free()
    cuda.DeviceAllocation().free()

    numba_cuda.select_device(0)
    numba_cuda.close()
    # if cxt == None:
        # pycuda.driver.Context.synchronize()
        # pycuda.driver.Context.pop()
    # else:
        # cxt.synchronize()
        # cxt.pop()

    # return endmembers
    return endmembers, abundances

# class LibPLSA(BC.BaseUnmixing):
    # '''
    # Libreria del algoritmo PLSA GPU
    # '''
    # def __init__(self, n_components=None, algorithm='plsa', max_iter=2000,
                # reg1=0.0, reg2=0.0, random_state=None):
        # self.n_components = n_components
        # self.components_ = None

    # def fit(self, data):
        # (p, X, denominators, theta, lamda, N, M) = generate_inputs(data, self.n_components)
        # initGPU()
        # pLSACUDA(lamda, theta, p, X, denominators, N, M, self.n_components, iters, r1, r2)

    # def transform(self, data):
        # return np.dot(data, self.components_.T)
            

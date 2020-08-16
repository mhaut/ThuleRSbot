# -*- coding: utf-8 -*-
from pycuda.compiler import SourceModule

kernels = SourceModule("""
   #include <math.h>

  __global__ void EStep(float* p, float* theta, float* lamda)
  {
     // Terminado
     __shared__ float denominator;
     unsigned idP = blockIdx.x*gridDim.y*blockDim.z + blockIdx.y*blockDim.z + threadIdx.z;
     unsigned idTheta = threadIdx.z*gridDim.y + blockIdx.y;
     unsigned idLamda = blockIdx.x*blockDim.z + threadIdx.z;
     if(threadIdx.z == 0) denominator = 0;
     __syncthreads();
     p[idP] = lamda[idLamda] * theta[idTheta];
     atomicAdd(&denominator, p[idP]);
     __syncthreads();
     if(denominator == 0)
          p[idP] = 0;
     else p[idP] /= denominator;
     __syncthreads();
  } 
  
    __global__ void EStepDPLSA(float* p, float* theta, float* lamda, int steps, int K)
  {
     // Terminado
     __shared__ float denominator;
     unsigned idTheta = threadIdx.z*gridDim.y + blockIdx.y;
     if(threadIdx.z == 0) denominator = 0;
     __syncthreads();
     for(int i = 0; (i < steps && (50 * i + threadIdx.z) < K); i++){
         p[blockIdx.x*gridDim.y*K + blockIdx.y*K + (50 * i + threadIdx.z)] = lamda[blockIdx.x*K + (50 * i + threadIdx.z)] * theta[idTheta];
         atomicAdd(&denominator, p[blockIdx.x*gridDim.y*K + blockIdx.y*K + (50 * i + threadIdx.z)]);
    }
     __syncthreads();
     for(int i = 0; (i < steps && (50 * i + threadIdx.z) < K); i++){
         if(denominator == 0)
              p[blockIdx.x*gridDim.y*K + blockIdx.y*K + (50 * i + threadIdx.z)] = 0;
         else p[blockIdx.x*gridDim.y*K + blockIdx.y*K + (50 * i + threadIdx.z)] /= denominator;
     }
     __syncthreads();
  } 
  
  
      __global__ void Theta1(float* theta, float* X, float* p, float r1, int steps, int N, int M, float* denominator)
  {
     // Terminado
     if(blockIdx.y == 0) denominator[blockIdx.z] = 0;
     unsigned idTheta = blockIdx.z*gridDim.y + blockIdx.y;
     if(threadIdx.x == 0) theta[idTheta] = 0;
     __syncthreads();
     for(int i = 0; (i < steps && (1024 * i + threadIdx.x) < N); i++){
             float valueaux = X[(1024 * i + threadIdx.x) * gridDim.y + blockIdx.y] * p[(1024 * i + threadIdx.x)*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z] - (1.0 * r1 / (1.0 * M));
             if(valueaux > 0)
                 atomicAdd(&theta[idTheta], valueaux);
             __syncthreads();
     }
  } 
     
  __global__ void Theta2(float* theta, float* denominator)
  {
     unsigned idTheta = blockIdx.z*blockDim.y + threadIdx.y;
     atomicAdd(&denominator[blockIdx.z], theta[idTheta]);
  } 
     
   
  __global__ void Theta3(float* theta, float* denominator)
  {
     unsigned idTheta = blockIdx.z*gridDim.y + blockIdx.y;
     if(denominator[blockIdx.z] == 0)
         theta[idTheta] = (1 / gridDim.y);
     else theta[idTheta] /= denominator[blockIdx.z];
  } 
     

  
    __global__ void Lamda(float* lamda, float* X, float* p, int K, float r2)
  {
     // Terminado
     __shared__ float denominator;
     unsigned idP = blockIdx.x*blockDim.y*gridDim.z + threadIdx.y*gridDim.z + blockIdx.z;
     unsigned idX = blockIdx.x * blockDim.y + threadIdx.y;
     unsigned idLamda = blockIdx.x*gridDim.z + blockIdx.z;
     if(threadIdx.z == 0){
             denominator = 0;
             lamda[idLamda] = 0;
     }
     __syncthreads();
     float valueaux = X[idX] * p[idP] - (1.0 * r2 / (1.0 * K));
     if(valueaux > 0)
         atomicAdd(&lamda[idLamda], valueaux);
     atomicAdd(&denominator, X[idX]);
     __syncthreads();
     if(threadIdx.y == 0){
         if(denominator == 0)
              lamda[idLamda] = (1 / blockDim.z);
         else lamda[idLamda] /= denominator;
     }
     __syncthreads();
  } 


  
  __global__ void LogLikelihood(double* returnedLogLike, float* X, float* theta, float* lamda)
  {
     // Terminado
     __shared__ double tmp;
     if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.z == 0) returnedLogLike[0] = 0;
     if(threadIdx.z == 0) tmp = 0;
     __syncthreads();
     for(int i = 0; i < blockDim.z; i++)   
         tmp += lamda[blockIdx.x*blockDim.z + i] * theta[i*gridDim.y + blockIdx.y];
     __syncthreads();
     if(tmp > 0 && threadIdx.z == 0){
        double val = X[blockIdx.x * gridDim.y + blockIdx.y] * log(tmp);
        atomicAdd(returnedLogLike, val);
     }
     __syncthreads();
  }
""")

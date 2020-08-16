# -*- coding: utf-8 -*-
#!/usr/bin/python3

from sklearn.metrics import mean_squared_error
import src.Algorithms._Base.BaseRestoration as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg
import numpy as np
import math
import cv2


class LibPansharpening():

    def __init__(self, typeSR, scale):
        self.scale = scale
        if typeSR == "bicubic":
            self.interpolation_type = cv2.INTER_CUBIC
        elif typeSR == "nearest":
            self.interpolation_type = cv2.INTER_NEAREST
        elif typeSR == "lanczos4":
            self.interpolation_type = cv2.INTER_LANCZOS4
        else:
            print("FATAL ERROR")
            exit()
        self.height = None
        self.width = None
        self.depth = None

    def fit(self, data):
        self.height, self.width, self.depth = data.shape

    def transform_score(self, data):
        newX, newY = data.shape[1] * (1 / float(self.scale)), data.shape[0] * (1 / float(self.scale))
        return cv2.resize(data, (int(newX), int(newY)), interpolation=self.interpolation_type)

    def transform(self, data):
        newX, newY = data.shape[1] * (float(self.scale)), data.shape[0] * (float(self.scale))
        return cv2.resize(data, (int(newX), int(newY)), interpolation=self.interpolation_type)

    def inverse_transform(self, data):
        return cv2.resize(data, (self.width, self.height), interpolation=self.interpolation_type)

    def score(self, image_true):
        image_pred = self.inverse_transform(self.transform_score(image_true))
        n_bands = image_true.shape[2]
        PSNR = np.zeros(n_bands)
        MSE = np.zeros(n_bands)
        mask = np.ones(n_bands)
        for k in range(n_bands):
            MSE[k] = mean_squared_error(image_true[:, :, k], image_pred[:, :, k])
            MAX_k = np.max(image_true[:, :, k])
            if MAX_k != 0 : PSNR[k] = 10.0 * np.log10(math.pow(MAX_k, 2) / MSE[k])
            else: mask[k] = 0.0
        psnr = PSNR.sum() / float(mask.sum())
        # mse = MSE.mean()
        return psnr  # , mse


class Pansharpening (BC.BaseRestoration):
    '''
    Class of algorithm Pansharpening
    '''

    def __init__(self, typeSR, scale, device='CPU'):
        self.model = LibPansharpening(typeSR=typeSR, scale=scale)
        self.device = device
        self.typeAlgorithm = enumTypeAlg.UNSUPERVISED

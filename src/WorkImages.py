'''
Auxiliary Method
'''

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from colormap import Colormap
import matplotlib.image as mpimg
import numpy as np
from osgeo import gdal
import scipy.io as sio
import os
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from PIL import Image


def hsi2rgb(result):
    orshape = [result.shape[0], result.shape[1], result.shape[2]]
    result = result.reshape(-1, result.shape[-1])
    num_components = 3
    if num_components < orshape[-1]:
        result = PCA(n_components=num_components).fit_transform(result)
        orshape[-1] = num_components
    result = MinMaxScaler(feature_range=(0, 255)).fit_transform(result).astype("uint8")
    result = result.reshape(orshape)
    return result


def show_image_mosaic(data, identifier, cmap=None, vmin=None, vmax=None, axis=False, IMGid=''):
    n_images = data.shape[0]    
    n_elements = int(np.ceil(np.sqrt(n_images)))
    # assert n_cols*n_rows == n_imagenes, "The n_images need be the same of n_cols*n_rows"
    all_imgs = np.ones((n_elements ** 2, data.shape[1], data.shape[2], data.shape[3]))
    if data.shape[3] == 3: all_imgs *= 255

    for posdat, dat in enumerate(data):
        all_imgs[posdat, :, :, :] = dat
    data = all_imgs.reshape(n_elements, n_elements, data.shape[1], data.shape[2], data.shape[3])
    del all_imgs

    _ , axarr = plt.subplots(n_elements, n_elements, gridspec_kw={'hspace':0.001, 'wspace':0})
    for i in range(n_elements):
        for j in range(n_elements):
            if data[i, j, :, :, :].shape[-1] == 1: 
                acdata = data[i, j, :, :, 0]
            else: 
                acdata = data[i, j, :, :, :]
            if n_elements == 1: 
                axarr.set_aspect('equal')
                axarr.imshow(acdata, cmap=cmap, vmin=vmin, vmax=vmax)
                if axis == False: axarr.axis('off')
            else: 
                axarr[i, j].set_aspect('equal')
                axarr[i, j].imshow(acdata, cmap=cmap, vmin=vmin, vmax=vmax)
                if axis == False: axarr[i, j].axis('off')
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('src/Consults/' + str(identifier) + '/image' + str(IMGid) + '.png', bbox_inches='tight', pad_inches=0.15)


def show_confusion_matrix(y_true, y_pred, IMGid=''):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0
    plt.imshow(cm_normalized, interpolation='nearest', cmap="gray_r")
    thresh = cm.max() / 2.
    for i, j in zip(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.0f'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    # plt.title("CONFUSION MATRIX")
    plt.colorbar()
    tick_marks = np.arange(len(y_true))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('src/Consults/' + str(identifier) + '/confusionMatrix' + str(IMGid) + '.png', bbox_inches='tight', pad_inches=0.15)


def show_single_image_CV2(data, identifier, IMGid=''):
    cv2.imwrite('src/Consults/' + str(identifier) + '/image' + str(IMGid) + '.png', data)

def resize_file_CV2(pathdata, resize):
    image = cv2.imread(pathdata).astype("uint8")
    ry = int(image.shape[0] / resize)
    rx = int(image.shape[1] / resize)
    image = cv2.resize(image, (rx, ry))
    cv2.imwrite(pathdata, image, [cv2.IMWRITE_PNG_COMPRESSION, 50])

def read_file(pathdata):
    extension = pathdata.strip().split(".")[-1]
    if extension == "mat":
        image = sio.loadmat(pathdata)
        # remove unnecesary fields
        keys = [a for a in image.keys() if a not in ['__version__', '__header__', '__globals__', 'abundances']]
        return image[keys[0]]
    elif extension == "tif":
        arys = []
        ds = gdal.Open(pathdata, gdal.GA_ReadOnly)
        for i in range(1, ds.RasterCount+1):
            arys.append(ds.GetRasterBand(i).ReadAsArray())
        return np.transpose(np.array(arys), (1,2,0))
    elif extension in ["png", "jpg", "jpeg"]:
        return cv2.imread(pathdata)
    else:
        raise Exception("The file is not supported")


def read_image_OthersOperation(pathdata, pathlabels=[], type_alg="spectral", preprocess_img="std"):
    image = read_file(pathdata)
    orshape = image.shape
    # preprocess
    imagePreprocesing = image.reshape(-1, image.shape[-1])
    if preprocess_img == "minmax": scaler = MinMaxScaler()
    elif preprocess_img == 'std':  scaler = StandardScaler()
    imagePreprocesing = scaler.fit_transform(imagePreprocesing)
    if not pathlabels: return image, imagePreprocesing, orshape
    else: labels = read_file(pathlabels)
    if type_alg == "spectral": labels = labels.reshape(-1)
    # remove label 0
    imagewithoutbg = imagePreprocesing[labels != 0]
    labelswithoutbg = labels[labels != 0] - 1
    return image, imagePreprocesing, imagewithoutbg, labels, labelswithoutbg, orshape


def split_image(train_percent, data, labels=[]):
    if len(labels) == 0: return train_test_split(data, test_size=1 - train_percent)
    else: return train_test_split(data, labels, test_size=1 - train_percent, stratify=labels)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def get_datas(pathFeature, preprocesing, trainPercent, pathLabels=[], typeDataset='Hyperspectral'):
    if typeDataset in ['Hyperspectral', 'Multispectral']:
        if(pathLabels):  # supervised
            # read file
            image, imagePreprocesing, imagewithoutbg, labels , labelswithoutbg, orshape = read_image_OthersOperation(pathdata=pathFeature, pathlabels=pathLabels, preprocess_img=preprocesing)
            
            if trainPercent < 1.0:
                X_train, X_test, Y_train, Y_test = split_image(trainPercent, imagewithoutbg, labels=labelswithoutbg)
            else:
                X_train = X_test = imagewithoutbg
                Y_train = Y_test = labelswithoutbg
            return image, imagePreprocesing, imagewithoutbg, labels , labelswithoutbg, orshape, X_train, X_test, Y_train, Y_test
        else:  # unsupervised
            # read the file
            image, imagePreprocesing, orshape = read_image_OthersOperation(pathFeature, pathLabels, preprocess_img=preprocesing)
            if trainPercent < 1.0:
                X_train, X_test = split_image(trainPercent, imagePreprocesing, labels=[])
            else:
                X_train = X_test = imagePreprocesing
            return image, imagePreprocesing, orshape, X_train, X_test
    
    elif typeDataset == 'RGB':
        classes = os.listdir(pathFeature)
        # filter_imgs = filter(lambda x: os.path.isfile(x), [])
        list_imgs = []; list_labels = []
        scaler = StandardScaler()
        for idclase, clase in enumerate(classes):
            images = [a for a in os.listdir(pathFeature + clase) if a.lower().endswith(".tif")]
            [list_labels.append(a) for a in [idclase] * len(images)]
            for pathim in images:
                acim = np.asarray(Image.open(pathFeature + clase + "/" + pathim).resize((224, 224), Image.ANTIALIAS))
                # scaler.partial_fit(acim.reshape(-1,acim.shape[-1]))
                list_imgs.append(acim)
        # orshape = list_imgs[0].shape
        # list_imgs = [scaler.transform(a.astype("float32").reshape(-1,acim.shape[-1])).reshape(orshape) for a in list_imgs]
        list_imgs = np.array(list_imgs)
        list_labels = np.array(list_labels)
        return unison_shuffled_copies(list_imgs, list_labels), len(classes)


def show_grafic(data, identifier, xlabel, ylabel, preprocesing=True, IMGid=''):
    if(preprocesing):
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(data)
    plt.savefig('src/Consults/' + str(identifier) + '/grafic' + str(IMGid) + '.png', bbox_inches='tight', pad_inches=0.15, dpi=200)


def show_changeDetection(data1, data2, result, rgb1, rgb2, messageId, colorMapDatas, colorMapResult, colorMapRGB, IMGid=''):
    _ , axarr = plt.subplots(1, 3, gridspec_kw={'hspace':0.001, 'wspace':0})
    axarr[0].imshow(data1, cmap=colorMapDatas, vmin=0)
    axarr[0].axis('off')
    axarr[1].imshow(data2, cmap=colorMapDatas, vmin=0)
    axarr[1].axis('off')
    axarr[2].imshow(result, cmap=colorMapResult, vmin=0)
    axarr[2].axis('off')
    plt.axis('off')
    plt.savefig('src/Consults/' + str(messageId) + '/Result' + IMGid + '.png', bbox_inches='tight', pad_inches=0.15)
    plt.clf()
    
    if len(rgb1) != 0 and len(rgb2) != 0 :
        plt.imshow(np.hstack((rgb1, rgb2)), cmap=colorMapRGB, vmin=0)
        plt.axis('off')
        plt.savefig('src/Consults/' + str(messageId) + '/RGB' + IMGid + '.png', bbox_inches='tight', pad_inches=0.15)

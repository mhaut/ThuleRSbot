# import
from copy import deepcopy
import gc
import numpy as np
import os
import scipy.io as sio
from src import WorkImages
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg
import importlib as im
import matplotlib.image as mpimg
import argparse
from src import Algorithms
from src import Tools
from sklearn.model_selection import train_test_split


def mainAlgorithm(listExecution, message, bot):
    '''
    Main method to execution algorithm 
    '''
    # create a dictionary
    switcherOp = {
        # classification
        'Classification' : algClassification,
        'DimensionalityReduction' : algDimensionalityReduction,
        'Unmixing'  : algUnmixing,
        'Restoration' : algRestoration,
        'ChangeDetection': algChangeDetection 
    }
    
    switcherOp.get(listExecution[message.chat.id]['Operation'])(listExecution, message, bot)

######################################################################
#############               CLASSIFICATION               #############
######################################################################


def algClassification(listExecution, message, bot):
    '''
    Classification algorithms
    ''' 
    messageOne = bot.send_message(message.chat.id, 'Loading...')
    
    alg = genericAlg(listExecution, message)
    
    pathFeature = listExecution[message.chat.id]['Upload'] + listExecution[message.chat.id]['Dataset']
    typeDataset = listExecution[message.chat.id]['DatasetTypeChosen']
    if typeDataset in ['Hyperspectral', 'Multispectral']:
        fname, ext = listExecution[message.chat.id]['Dataset'].split(".")
        preprocesing = listExecution[message.chat.id]['Parameters']['preprocesing']['value']
    trainPercent = float(listExecution[message.chat.id]['Parameters']['trainPercent']['value'])
    
    # supervised
    if(alg.typeAlgorithm == enumTypeAlg.SUPERVISED):
        if typeDataset in ['Hyperspectral', 'Multispectral']:
            pathLabels = listExecution[message.chat.id]['Upload'] + fname + '_gt.' + ext
        if(typeDataset in ['Hyperspectral', 'Multispectral'] and not os.path.isfile(pathLabels)):
            bot.send_message(message.chat.id, 'No exits labels file')
            raise Exception('No exits labels file')
        
        # read datas
        if typeDataset in ['Hyperspectral', 'Multispectral']:
            image, imagePreprocesing, _, labels , _, orshape, X_train, X_test, Y_train, Y_test = WorkImages.get_datas(pathFeature=pathFeature, pathLabels=pathLabels, preprocesing=preprocesing, trainPercent=trainPercent, typeDataset=typeDataset)
        elif  typeDataset == 'RGB':    
            (X, y), n_classes = WorkImages.get_datas(pathFeature=pathFeature, pathLabels=[], preprocesing=None, trainPercent=trainPercent, typeDataset=typeDataset)
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=(1 - trainPercent), random_state=0)  # AVISO random
            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
            X_train /= 255
            X_test /= 255
        
        # execution
        alg.train_algorithm(deepcopy(X_train), deepcopy(Y_train))
        Y_pred, _ = alg.test_algorithm(deepcopy(X_test))
        precision = alg.score(deepcopy(X_test), deepcopy(Y_test))
        
        if typeDataset in ['Hyperspectral', 'Multispectral']:
            result, _ = alg.test_algorithm(deepcopy(imagePreprocesing))
            result += 1  # background: label 0
            # GT
            WorkImages.show_image_mosaic(labels.reshape(-1, orshape[0], orshape[1], 1), message.chat.id, cmap='jet', IMGid="GT", vmin=0)
            
    # unsupervised
    elif(alg.typeAlgorithm == enumTypeAlg.UNSUPERVISED):
        # read datas
        pathLabels = []
        image, imagePreprocesing, orshape, X_train, X_test = WorkImages.get_datas(pathFeature=pathFeature, pathLabels=pathLabels, preprocesing=preprocesing, trainPercent=trainPercent, typeDataset=typeDataset)

        # execution
        alg.train_algorithm(deepcopy(X_train))
        alg.test_algorithm(deepcopy(X_test))
        precision = None
        result, _ = alg.test_algorithm(deepcopy(imagePreprocesing))
    
    # other
    else:
        bot.send_message(message.chat.id, "This operation does not exist")
        raise Exception("This operation does not exist")
    # save precision
    listExecution[message.chat.id]['resultPrecision'] = precision
    bot.delete_message(message.chat.id, messageOne.message_id)
    
    if typeDataset in ['Hyperspectral', 'Multispectral']:
        # save original image (hpi) as rgbs
        pathRGB = listExecution[message.chat.id]['Upload'] + fname + '_rgb.' + ext
        if not os.path.exists(pathRGB):
            imageOriginalRGB = WorkImages.hsi2rgb(image)
        else:
            imageOriginalRGB = WorkImages.read_file(pathRGB)
            
        WorkImages.show_single_image_CV2(imageOriginalRGB, message.chat.id, IMGid='Original')
        
        result = result.T.reshape(-1, orshape[0], orshape[1], 1)
        WorkImages.show_image_mosaic(result, message.chat.id, cmap='jet', vmin=0, IMGid="Result")
    
        initPath = 'src/Consults/' + str(message.chat.id) + "/"
        if(alg.typeAlgorithm == enumTypeAlg.SUPERVISED):
            bot.send_photo(chat_id=message.chat.id, photo=open(initPath + 'imageGT.png', 'rb'), caption='GT')
        
        size = os.stat(initPath + 'imageOriginal.png').st_size / 1e6 / 5
        if size > 1:
            WorkImages.resize_file_CV2(initPath + 'imageOriginal.png', size)
        bot.send_photo(chat_id=message.chat.id, photo=open(initPath + 'imageOriginal.png', 'rb'), caption='Original Image')
        bot.send_photo(chat_id=message.chat.id, photo=open(initPath + 'imageResult.png', 'rb'), caption='{} algorithm result. Precision OA: {}'.format(listExecution[message.chat.id]['Algorithm'], precision))
    elif  typeDataset == 'RGB':
        WorkImages.show_confusion_matrix(Y_test, Y_pred)
        bot.send_photo(chat_id=message.chat.id, photo=open(initPath + 'confusionMatrix.png', 'rb'), caption='{} algorithm result. Precision OA: {}'.format(listExecution[message.chat.id]['Algorithm'], precision))
    # remove alg
    del alg; gc.collect()


######################################################################
#############     Dimensionality Reduction               #############
######################################################################
def algDimensionalityReduction(listExecution, message, bot):
    '''
    Dimensionality Reduction algorithms
    ''' 

    messageOne = bot.send_message(message.chat.id, 'Loading...')
    
    alg = genericAlg(listExecution, message)
    
    pathFeature = listExecution[message.chat.id]['Upload'] + listExecution[message.chat.id]['Dataset']
    typeDataset = listExecution[message.chat.id]['DatasetTypeChosen']
    if typeDataset in ['Hyperspectral', 'Multispectral']:
        fname, ext = listExecution[message.chat.id]['Dataset'].split(".")
        preprocesing = listExecution[message.chat.id]['Parameters']['preprocesing']['value']
    trainPercent = float(listExecution[message.chat.id]['Parameters']['trainPercent']['value'])
    
    # supervised
    if(alg.typeAlgorithm == enumTypeAlg.SUPERVISED):
        pathLabels = listExecution[message.chat.id]['Upload'] + fname + '_gt.' + ext
        if(os.path.isfile(pathLabels)):
            # read datas
            image, imagePreprocesing, _, labels , _, orshape, X_train, X_test, Y_train, Y_test = WorkImages.get_datas(pathFeature=pathFeature, pathLabels=pathLabels, preprocesing=preprocesing, trainPercent=trainPercent, typeDataset=typeDataset)
            
            # execution
            alg.train_algorithm(deepcopy(X_train), deepcopy(Y_train))
            alg.test_algorithm(deepcopy(X_test))
            precision = alg.score(deepcopy(X_test), deepcopy(Y_test))
            
            result, _ = alg.test_algorithm(deepcopy(imagePreprocesing))
            WorkImages.show_image_mosaic(labels.reshape(-1, orshape[0], orshape[1], 1), message.chat.id, cmap='jet', IMGid="GT", vmin=0)
        else:
            bot.send_message(message.chat.id, 'No exits labels file')
            raise Exception('No exits labels file')
    
    # unsupervised
    elif(alg.typeAlgorithm == enumTypeAlg.UNSUPERVISED):
        # read datas
        pathLabels = []
        image, imagePreprocesing, orshape, X_train, X_test = WorkImages.get_datas(pathFeature=pathFeature, pathLabels=pathLabels, preprocesing=preprocesing, trainPercent=trainPercent, typeDataset=typeDataset)
        
        # execution
        alg.train_algorithm(deepcopy(X_train))
        alg.test_algorithm(deepcopy(X_test))
        precision = alg.score(deepcopy(X_test))
        result, _ = alg.test_algorithm(deepcopy(imagePreprocesing))
    
    # other
    else:
        bot.send_message(message.chat.id, "This operation does not exist")
        raise Exception("This operation does not exist")
    
    # save precision
    listExecution[message.chat.id]['resultPrecision'] = precision
    
    # save original image (hpi) as rgbs
    pathRGB = listExecution[message.chat.id]['Upload'] + fname + '_rgb.' + ext
    if not os.path.exists(pathRGB):
        imageOriginalRGB = WorkImages.hsi2rgb(image)
    else:
        imageOriginalRGB = WorkImages.read_file(pathRGB)
    WorkImages.show_single_image_CV2(imageOriginalRGB, message.chat.id, IMGid='Original')
    
    bot.delete_message(message.chat.id, messageOne.message_id)
    
    result = result.T.reshape(-1, orshape[0], orshape[1], 1)
    WorkImages.show_image_mosaic(result, message.chat.id, cmap='jet', IMGid="Result")
    
    if(alg.typeAlgorithm == enumTypeAlg.SUPERVISED):
        bot.send_photo(chat_id=message.chat.id, photo=open('src/Consults/' + str(message.chat.id) + '/imageGT.png', 'rb'), caption='GT')
    bot.send_photo(chat_id=message.chat.id, photo=open('src/Consults/' + str(message.chat.id) + '/imageOriginal.png', 'rb'), caption='Original Image')
    bot.send_photo(chat_id=message.chat.id, photo=open('src/Consults/' + str(message.chat.id) + '/imageResult.png', 'rb'), caption='{} algorithm result. Precision OA: {}'.format(listExecution[message.chat.id]['Algorithm'], precision))
    
    # remove alg
    del alg; gc.collect()


######################################################################
#############              Restoration                   #############
######################################################################
def algRestoration(listExecution, message, bot):
    '''
    Restoration algorithms 
    '''
    
    messageOne = bot.send_message(message.chat.id, 'Loading...')
    
    alg = genericAlg(listExecution, message)

    pathFeature = listExecution[message.chat.id]['Upload'] + listExecution[message.chat.id]['Dataset']
    fname, ext = listExecution[message.chat.id]['Dataset'].split(".")
    pathRGB = listExecution[message.chat.id]['Upload'] + fname + '_rgb.' + ext
       
    # read file
    if not os.path.exists(pathRGB):
        image = WorkImages.read_file(pathFeature)
        imageOriginalRGB = image
        if image.shape[-1] > 3: imageOriginalRGB = WorkImages.hsi2rgb(image)
    else:
        imageOriginalRGB = WorkImages.read_file(pathRGB)
    
    # supervised
    if(alg.typeAlgorithm == enumTypeAlg.SUPERVISED):
        # TODO when add algorithm typeAlgorithm
        pass
    
    # unsupervised
    elif(alg.typeAlgorithm == enumTypeAlg.UNSUPERVISED):
        # execution
        # if exist RGB
        if os.path.exists(pathRGB):
            alg.train_algorithm(deepcopy(imageOriginalRGB))
            result, _ = alg.test_algorithm(deepcopy(imageOriginalRGB))
            psnr = alg.score(deepcopy(imageOriginalRGB))
        else:
            alg.train_algorithm(deepcopy(image))
            result, _ = alg.test_algorithm(deepcopy(image))
            psnr = alg.score(deepcopy(image))
            # postprocesing
            if image.shape[-1] > 3: result = WorkImages.hsi2rgb(result)
    
    # other
    else:
        bot.send_message(message.chat.id, "This operation does not exist")
        raise Exception("This operation does not exist")
    
    # save precision
    listExecution[message.chat.id]['resultPrecision'] = psnr
    
    # show image
    bot.delete_message(message.chat.id, messageOne.message_id)
    WorkImages.show_single_image_CV2(result, message.chat.id, IMGid='Result')
    WorkImages.show_single_image_CV2(imageOriginalRGB, message.chat.id, IMGid='Original')
    
    bot.send_photo(chat_id=message.chat.id, photo=open('src/Consults/' + str(message.chat.id) + '/imageOriginal.png', 'rb'), caption='Original Image')
    bot.send_photo(chat_id=message.chat.id, photo=open('src/Consults/' + str(message.chat.id) + '/imageResult.png', 'rb'), caption='{} algorithm result. Precision: psnr {}'.format(listExecution[message.chat.id]['Algorithm'], psnr))
    
    # remove alg
    del alg; gc.collect()


######################################################################
#############                 Unmixing                   #############
######################################################################
def algUnmixing(listExecution, message, bot):
    '''
    Unmixing algorithms
    '''
    
    messageOne = bot.send_message(message.chat.id, 'Loading...')
    
    alg = genericAlg(listExecution, message)
    
    # read file
    pathImage = listExecution[message.chat.id]['Upload'] + listExecution[message.chat.id]['Dataset']
    if listExecution[message.chat.id]['Dataset'].endswith('.mat'):
        file = sio.loadmat(pathImage)
        image = file['Y'].T
        nCol = int(file['nCol'][0])
        nRow = int(file['nRow'][0])
    else:
        file = mpimg.imread(pathImage)
        nRow = int(file.shape[1])
        nCol = int(file.shape[0])
        image = np.transpose(file, axes=(1, 0, 2)).reshape(-1, file.shape[2])
    
    fname, ext = listExecution[message.chat.id]['Dataset'].split(".")
    pathEnd = listExecution[message.chat.id]['Upload'] + fname + '_end.' + ext
     
    # supervised
    if(alg.typeAlgorithm == enumTypeAlg.SUPERVISED):
        if(os.path.isfile(pathEnd)):
            end = sio.loadmat(pathEnd)
            signatureEndmember = end['M']
            
            # execution
            alg.train_algorithm(deepcopy(image), deepcopy(signatureEndmember))
            result, _ = alg.test_algorithm(deepcopy(image))
            rmse = alg.score(data=deepcopy(result), labels=deepcopy(pathEnd))
        else:
            bot.send_message(message.chat.id, 'No exits End file')
            raise Exception('No exits End file')
        
    # unsupervised
    elif(alg.typeAlgorithm == enumTypeAlg.UNSUPERVISED):
        
        # execution
        alg.train_algorithm(deepcopy(image))
        result, _ = alg.test_algorithm(deepcopy(image))
        signatureEndmember = alg.model.components_.T
        if os.path.exists(pathEnd):
            rmse = alg.score(data=deepcopy(result), labels=deepcopy(pathEnd))
            # sad = alg.scoreSad(signatureEndmember, pathImage, pathEnd)
        else:
            rmse = None
    
    # other
    else:
        bot.send_message(message.chat.id, "This operation does not exist")
        raise Exception("This operation does not exist")
        
    # save precision
    listExecution[message.chat.id]['resultPrecision'] = rmse
    
    bot.delete_message(message.chat.id, messageOne.message_id)
    
    # save original image (hpi) as rgb
    pathRGB = listExecution[message.chat.id]['Upload'] + fname + '_rgb.' + ext
    
    if not os.path.exists(pathRGB):
        imageOriginalRGB = np.transpose(WorkImages.hsi2rgb(image.reshape(nRow, nCol, -1)), axes=(1, 0, 2))
    else:
        imageOriginalRGB = WorkImages.read_file(pathRGB)
        
    WorkImages.show_single_image_CV2(imageOriginalRGB, message.chat.id, IMGid='Original')
    bot.send_photo(chat_id=message.chat.id, photo=open('src/Consults/' + str(message.chat.id) + '/imageOriginal.png', 'rb'), caption='Original Image')
    
    # show endmembers
    WorkImages.show_grafic(signatureEndmember, message.chat.id, xlabel='Bandas', ylabel='Endmembers', IMGid='Endmembers')
    bot.send_photo(chat_id=message.chat.id, photo=open('src/Consults/' + str(message.chat.id) + '/graficEndmembers.png', 'rb'), caption='{} algorithm result. Endmembers'.format(listExecution[message.chat.id]['Algorithm']))
    
    # show image
    result = np.transpose(result.T.reshape(-1, nRow, nCol, 1), axes=(0, 2, 1, 3))
    WorkImages.show_image_mosaic(result, message.chat.id, cmap='gray', vmin=0, vmax=1, IMGid='Result')
    bot.send_photo(chat_id=message.chat.id, photo=open('src/Consults/' + str(message.chat.id) + '/imageResult.png', 'rb'), caption='{} algorithm result. Abundances. Precision: rmse {}'.format(listExecution[message.chat.id]['Algorithm'], rmse))
    
    # remove alg
    del alg; gc.collect()


def algChangeDetection(listExecution, message, bot):
    messageOne = bot.send_message(message.chat.id, 'Loading...')
    alg = genericAlg(listExecution, message)
    
    # read images
    pathImage = listExecution[message.chat.id]['Upload'] + listExecution[message.chat.id]['Dataset']
    listImages = Tools.lsDataset(pathImage, listExecution[message.chat.id]['Extension'])
    
    data1 = WorkImages.read_file(pathImage + listImages[0])
    data2 = WorkImages.read_file(pathImage + listImages[1])
    fname1, ext1 = listImages[0].split(".")
    fname2, ext2 = listImages[1].split(".")
    
    if alg.typeAlgorithm == enumTypeAlg.SUPERVISED:
        labels1 = WorkImages.read_file(pathImage + fname1 + '_gt.' + ext1)[:, :, 0]
        labels2 = WorkImages.read_file(pathImage + fname2 + '_gt.' + ext2)[:, :, 0]
        result, _ = alg.execution(data1, data2, labels1, labels2)
        
    elif alg.typeAlgorithm == enumTypeAlg.UNSUPERVISED:
        result, _ = alg.execution(data1, data2)
    else:
        bot.send_message(message.chat.id, "This operation does not exist")
        raise Exception("This operation does not exist")
    
    if os.path.exists(pathImage + fname1 + '_rgb.' + ext1) and  os.path.exists(pathImage + fname2 + '_rgb.' + ext2):
        data1RGB = WorkImages.read_file(pathImage + fname1 + '_rgb.' + ext1)
        data2RGB = WorkImages.read_file(pathImage + fname2 + '_rgb.' + ext2)
    else:
        data1RGB = []
        data2RGB = []
    
    # save precision
    listExecution[message.chat.id]['resultPrecision'] = None
    
    bot.delete_message(message.chat.id, messageOne.message_id)
    WorkImages.show_changeDetection(result[0], result[1], result[2], data1RGB, data2RGB, message.chat.id,
                                    alg.model.colorMapDatas, alg.model.colorMapResult, alg.model.colorMapRGB)
    
    bot.send_photo(chat_id=message.chat.id, photo=open('src/Consults/' + str(message.chat.id) + '/RGB.png', 'rb'), caption='RGB')
    bot.send_photo(chat_id=message.chat.id, photo=open('src/Consults/' + str(message.chat.id) + '/Result.png', 'rb'), caption='Result')

    
######################################################################
#############           Get Especific Algorithm          #############
######################################################################
def genericAlg(listExecution, message):    
    # import
    path = "src.Algorithms." + listExecution[message.chat.id]["Operation"] + "." + listExecution[message.chat.id]["Type"] + "." + listExecution[message.chat.id]["Algorithm"]  
    cls = getattr(im.import_module(path), listExecution[message.chat.id]["Algorithm"])
    
    # parameters
    stringParameter = ""
    parser = argparse.ArgumentParser()
    if listExecution[message.chat.id]['Parameters']:  # si hay parametros
        for k, v in listExecution[message.chat.id]['Parameters'].items():
            if k in Algorithms.FileParametersDefault[listExecution[message.chat.id]["Operation"]][listExecution[message.chat.id]['Type']][listExecution[message.chat.id]["Algorithm"]]['parameters']:
                if v['type'] == 'int':
                    parser.add_argument('-' + str(k), type=int, action="store", dest=str(k))
                elif v['type'] == 'float':
                    parser.add_argument('-' + str(k), type=float, action="store", dest=str(k))
                else:
                    parser.add_argument('-' + str(k), type=str, action="store", dest=str(k))
                
                stringParameter += '-' + str(k) + " " + str(v['value']) + " "

    return cls(**vars(parser.parse_args(stringParameter.split())))

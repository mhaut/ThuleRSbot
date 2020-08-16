#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Auxiliary Method
'''
# Import
from telebot import types
import os
import glob
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


def createButton(arrayButtons):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    
    for pos in range(0, len(arrayButtons) - 1, 2):
        markup.row(str(arrayButtons[pos]), str(arrayButtons[pos + 1]))
    
    if len(arrayButtons) % 2 != 0:
        markup.row(str(arrayButtons[-1]))

    return markup


def myls(path, typels="dir", extensions=[], filterfunc=[], onlyfname=False, returnExtension=True):
    assert typels in ["file", "dir"]
    if extensions:
        filesdir = sum([glob.glob(path + "*" + ex) for ex in extensions], [])  
    else:
        filesdir = glob.glob(path + "*")
    listdeny = sum([glob.glob(path + ff) for ff in filterfunc], [])
    filterfunc = os.path.isdir if typels == "dir" else os.path.isfile
    filesdir = [a for a in filesdir if a not in listdeny and filterfunc(a)]
    if onlyfname: 
        if returnExtension:
            filesdir = [os.path.basename(a) for a in filesdir]
        else:
            filesdir = [os.path.basename(a).split(".")[0] for a in filesdir]
    return sorted(filesdir)


def lsFolder(pathDir, excludes=[]):
    return myls(pathDir, typels="dir", filterfunc=["_*"], onlyfname=True)

    
def lsArch_py(pathDir, excludes=[]):
    return myls(pathDir, extensions=[".py"], typels="file", filterfunc=["_*"], onlyfname=True, returnExtension=False)


def lsDataset(pathDir, listExtension, excludes=[]):
    return myls(pathDir, extensions=listExtension, typels="file", filterfunc=["_*", "*_gt*", "*_end*", "*_rgb*"], onlyfname=True)


def lsArch_png(pathDir, excludes=[]):
    return myls(pathDir, extensions=[".png"], typels="file", filterfunc=["_*"], onlyfname=False)


def saveLog(stringSave, path):
    f = open (path + 'log.txt', 'a')
    f.write(stringSave + '\n\n')
    f.close()


def get_gpus_avaiables():
    from keras import backend as K
    K.tensorflow_backend._get_available_gpus()
    
    
def create_zip(pathLisElements, pathSave, name):
    import zipfile
    try:
        compression = zipfile.ZIP_DEFLATED
    except:
        compression = zipfile.ZIP_STORED
    
    zf = zipfile.ZipFile(pathSave + name, mode="w")
    
    try:
        for element in pathLisElements:
            nameFile = element.split('/')[-1]
            zf.write(element, nameFile, compress_type=compression)

    finally:
        zf.close()

        
def _walk(path, depth):
    """Recursively list files and directories up to a certain depth"""
    depth -= 1
    with os.scandir(path) as p:
        for entry in p:
            yield entry.path
            if entry.is_dir() and depth > 0:
                yield from _walk(entry.path, depth)


def remove_keys(obj, rubbish):
    if isinstance(obj, dict):
        obj = {
            key: remove_keys(value, rubbish) 
            for key, value in obj.items()
            if key not in rubbish}
    elif isinstance(obj, list):
        obj = [remove_keys(item, rubbish)
                  for item in obj
                  if item not in rubbish]
    return obj


def search_in_depth(obj, algs_comp):
    for k, v in obj.items():
        if k == "AllowedDatasetType":
            for pos, val in enumerate(v):
                algs_comp[val] += 1
            return algs_comp
        elif isinstance(v, dict):
            search_in_depth(v, algs_comp)
        # else:
    return algs_comp


def findOptionAvailable(listOption, dict, typeDataset):
    options = []
    for op in listOption:
        algs_comp = {'Hyperspectral':0, 'Multispectral':0, 'RGB':0}
        if dict[op] != None:
            tengoalgs = search_in_depth(dict[op], algs_comp)
        if algs_comp[typeDataset] > 0:
            options.append(op)
    return options


def SendEmail(bot, password, from_addrs, to_addrs, subject, body, message, zippath, zipname):
        '''
        Method to send an email
        '''
        messageOne = bot.send_message(message.chat.id, 'Sending')
        # create message object instance
        msg = MIMEMultipart()
        
        # setup the parameters of the message
        msg['From'] = from_addrs
        msg['To'] = to_addrs
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))
        archivo_adjunto = open(zippath, 'rb')
        adjunto_MIME = MIMEBase('application', 'octet-stream')
        adjunto_MIME.set_payload((archivo_adjunto).read())
        encoders.encode_base64(adjunto_MIME)
        adjunto_MIME.add_header('Content-Disposition', "attachment; filename= %s" % zipname)
        msg.attach(adjunto_MIME)
    
        # send email
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.starttls()
        
        try:
            server.login(msg['From'], password)
        except:
            bot.send_message(message.chat.id, "Incorrect authentication")
            server.close()
            raise Exception('Email')
        
        try:
            server.sendmail(msg['From'], msg['To'], msg.as_string())
        except:
            bot.send_message("The email could not be sent")
            server.close()
            raise Exception('Email')
    
        server.quit()
        
        bot.delete_message(message.chat.id, messageOne.message_id)
        bot.send_message(message.chat.id, 'The email was sent successfully')
        

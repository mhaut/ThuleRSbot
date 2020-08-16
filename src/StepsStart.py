#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import
import datetime, time, threading, os, sys, threading
from src import Tools
from src import Algorithms
from src.MainAlgorithms import mainAlgorithm
from copy import deepcopy
from shutil import rmtree
import scipy.io as sio
import numpy as np

# Sentinel
from sentinelhub import SHConfig, WmsRequest, CRS, BBox, DataSource
from eolearn.core.eoworkflow import LinearWorkflow
from eolearn.io import S2L1CWCSInput, L8L1CWCSInput, AddSen2CorClassificationFeature


class StepsStart(object):
    '''
    Start Command Process Class
    '''

    def __init__(self, bot, listExecution, queueExec=None):
        self.bot = bot
        self.listExecution = listExecution
        # wait for messages
        self.bot.set_update_listener(self.listener)
        self.lock = threading.Lock()
        self.userInWait = 0
        self.numThread = Algorithms.FileConfig["Telegram"]["num_Threads"]
        self.queueExec = queueExec

    def StepZero(self, message, show_help=False):
        '''
        Method to select type dataset
        '''
        # clear list of execution
        if self.listExecution.get(message.chat.id): del self.listExecution[message.chat.id]
        self.listExecution[message.chat.id] = {}
        # Current time
        self.listExecution[message.chat.id]['Time'] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.listExecution[message.chat.id]['CurrentStep'] = 0
        typeDataset = ['Hyperspectral', 'Multispectral', 'RGB']
        # buttons
        markup = Tools.createButton(typeDataset)
        if show_help: messageR = self.bot.send_message(message.chat.id, "Please, follow the provided instructions", reply_markup=markup)
        else: messageR = self.bot.send_message(message.chat.id, "What type of dataset do you want to use?", reply_markup=markup)
        # next step(callback)
        self.bot.register_next_step_handler(messageR, self.AnswerStepZero, typeDataset)

    def AnswerStepZero(self, message, typeDataset):
        '''
        Method that manages the response provided by the user in StepZero
        '''
        if not self.stepStartCommands(message):
            # correct option?
            if message.text in typeDataset:
                # save operation
                self.listExecution[message.chat.id]['DatasetTypeChosen'] = message.text
                self.StepOne(message)
            else:
                self.bot.send_message(message.chat.id, "ERROR: Select a correct option please")
                self.StepZero(message)
        else: self.StepZero(message, show_help=True)

    def StepOne(self, message, show_help=False):
        '''
        Method to select operation
        '''
        self.listExecution[message.chat.id]['CurrentStep'] = 1
        
        # operations available
        operations = Tools.findOptionAvailable(Tools.lsFolder('src/Algorithms/'), Algorithms.FileParametersDefault, self.listExecution[message.chat.id]['DatasetTypeChosen'])
        operations.append('Back')
        # buttons
        markup = Tools.createButton(operations)
        if show_help: messageR = self.bot.send_message(message.chat.id, "Please, follow the provided instructions", reply_markup=markup)
        else: messageR = self.bot.send_message(message.chat.id, "What do you want to do?", reply_markup=markup)
        # next step(callback)
        self.bot.register_next_step_handler(messageR, self.AnswerStepOne, operations)

    def AnswerStepOne(self, message, operations):
        '''
        Method that manages the response provided by the user in StepOne
        '''
        if not self.stepStartCommands(message):
            # correct option?
            if message.text in operations:
                if message.text == 'Back':
                    self.StepZero(message)
                else:
                    # save operation
                    self.listExecution[message.chat.id]['Operation'] = message.text
                    self.StepTwo(message)
            else:
                self.bot.send_message(message.chat.id, "ERROR: Select a correct option please")
                self.StepOne(message)
        else: self.StepOne(message, show_help=True)

    def StepTwo(self, message, show_help=False):
        '''
        Method to select type
        '''
        self.listExecution[message.chat.id]['CurrentStep'] = 2
        # types available 
        typeAllAlg = Tools.lsFolder('src/Algorithms/' + str(self.listExecution[message.chat.id]['Operation']) + '/')
        dict = Algorithms.FileParametersDefault[self.listExecution[message.chat.id]['Operation']]
        
        typeAlg = Tools.findOptionAvailable(typeAllAlg, dict, self.listExecution[message.chat.id]['DatasetTypeChosen'])
        typeAlg.append('Back')
        # buttons
        markup = Tools.createButton(typeAlg)
        if show_help: messageR = self.bot.send_message(message.chat.id, "Please, follow the provided instructions", reply_markup=markup)
        else: messageR = self.bot.send_message(message.chat.id, "Choose the type", reply_markup=markup)
        # next step (callback)
        self.bot.register_next_step_handler(messageR, self.AnswerStepTwo, typeAlg)

    def AnswerStepTwo(self, message, typeAlg):
        '''
        Method that manages the response provided by the user in StepThree
        '''
        if not self.stepStartCommands(message):
            # correct option?
            if message.text in typeAlg:
                if(message.text == 'Back'): self.StepOne(message)
                else:
                    # save type
                    self.listExecution[message.chat.id]['Type'] = message.text
                    self.StepThree(message)
            else:
                self.bot.send_message(message.chat.id, "ERROR: Select a correct option please")
                self.StepTwo(message)
        else: self.StepTwo(message, show_help=True)

    def StepThree(self, message, show_help=False):
        '''
        Method to select algorithm
        '''
        self.listExecution[message.chat.id]['CurrentStep'] = 3
        # search all algorithms
        allAlgorithms = Tools.lsArch_py('src/Algorithms/' + str(self.listExecution[message.chat.id]['Operation']) + "/" + str(self.listExecution[message.chat.id]['Type']) + "/")
        
        algorithms = []
        # filterAlgorithm
        for alg in allAlgorithms:
            if(self.listExecution[message.chat.id]['DatasetTypeChosen'] in Algorithms.FileParametersDefault[self.listExecution[message.chat.id]['Operation']][self.listExecution[message.chat.id]['Type']][alg]['info']['AllowedDatasetType']):
                algorithms.append(alg)
        algorithms.append('Back')
        
        # buttons
        markup = Tools.createButton(algorithms)
        if show_help: messageR = self.bot.send_message(message.chat.id, "Please, follow the provided instructions", reply_markup=markup)
        else: messageR = self.bot.send_message(message.chat.id, "Which algorithm do you want to use?", reply_markup=markup)
        self.bot.register_next_step_handler(messageR, self.AnswerStepThree, algorithms)

    def AnswerStepThree(self, message, algorithms):
        '''
        Method that manages the response provided by the user in StepThree
        '''
        try:
            if not self.stepStartCommands(message):
                # correct option?
                if message.text in algorithms:
                    if(message.text == 'Back'): self.StepTwo(message)
                    else:    
                        # save algorithm
                        self.listExecution[message.chat.id]['Algorithm'] = message.text
                        operation = self.listExecution[message.chat.id]['Operation']
                        typealg = self.listExecution[message.chat.id]['Type']
                        algorithm = self.listExecution[message.chat.id]['Algorithm']
                        
                        # clear parameters
                        if self.listExecution[message.chat.id].get('Parameters'): del self.listExecution[message.chat.id]['Parameters']
                        self.listExecution[message.chat.id]['Parameters'] = {}
                        
                        # are there any parameters?
                        if(algorithm in Algorithms.FileParametersDefault[operation][typealg] and Algorithms.FileParametersDefault[operation][typealg][algorithm]):
                            # save parameters
                            self.listExecution[message.chat.id]['Parameters'] = deepcopy(Algorithms.FileParametersDefault[operation][typealg][algorithm]['parameters'])
                            self.listExecution[message.chat.id]['singleOrMultipleImage'] = Algorithms.FileParametersDefault[operation][typealg][algorithm]['info']['singleOrMultipleImage']
                        else:
                            raise Exception('Error to read file yaml')
                        
                        # save extensions
                        self.listExecution[message.chat.id]['Extension'] = deepcopy(Algorithms.FileDatasets[operation][self.listExecution[message.chat.id]['DatasetTypeChosen']][self.listExecution[message.chat.id]['singleOrMultipleImage']]['extensions'])
                        
                        if 'trainPercent' in Algorithms.FileDatasets[operation][self.listExecution[message.chat.id]['DatasetTypeChosen']][self.listExecution[message.chat.id]['singleOrMultipleImage']]:
                            self.listExecution[message.chat.id]['Parameters']['trainPercent'] = deepcopy(Algorithms.FileDatasets[operation][self.listExecution[message.chat.id]['DatasetTypeChosen']][self.listExecution[message.chat.id]['singleOrMultipleImage']]['trainPercent'])
                        if 'preprocesing' in Algorithms.FileDatasets[operation][self.listExecution[message.chat.id]['DatasetTypeChosen']][self.listExecution[message.chat.id]['singleOrMultipleImage']]:
                            self.listExecution[message.chat.id]['Parameters']['preprocesing'] = deepcopy(Algorithms.FileDatasets[operation][self.listExecution[message.chat.id]['DatasetTypeChosen']][self.listExecution[message.chat.id]['singleOrMultipleImage']]['preprocesing'])
                        # next step
                        self.StepFour(message)
                else:
                    self.bot.send_message(message.chat.id, "ERROR: Select a correct option please")
                    self.StepThree(message)
            else: self.StepThree(message, show_help=True)
        except:
            self.bot.send_message(message.chat.id, "Error to read file yaml")
            self.StepThree(message)

    def StepFour(self, message, show_help=False):
        '''
        Method to select dataset
        '''
        self.listExecution[message.chat.id]['CurrentStep'] = 4
        
        singlemultipleimage = Algorithms.FileParametersDefault[self.listExecution[message.chat.id]['Operation']][self.listExecution[message.chat.id]['Type']][self.listExecution[message.chat.id]['Algorithm']]['info']['singleOrMultipleImage']
        
        buttomVector = []
        if self.listExecution[message.chat.id]['Operation'] != 'ChangeDetection':
            if singlemultipleimage in ['single', 'both']:
                buttomVector += ["Public"] 
                buttomVector += ["Private", "Upload"]
                if self.listExecution[message.chat.id]['DatasetTypeChosen'] == 'Multispectral':
                    buttomVector += ["Sentinel-Hub"]
            if singlemultipleimage in ['multiple', 'both']:
                if self.listExecution[message.chat.id]['DatasetTypeChosen'] == 'RGB':
                    buttomVector += ["Unmerced"]
        else:
            buttomVector += ["Sentinel-Hub", "ChangeDetection"]
        buttomVector += ["Default", "Back"]
        # buttons
        markup = Tools.createButton(buttomVector)
        if show_help:
            messageR = self.bot.send_message(message.chat.id, "Please, follow the provided instructions", reply_markup=markup)
        else:
            messageR = self.bot.send_message(message.chat.id, "Which dataset do you want to use?", reply_markup=markup)
        self.bot.register_next_step_handler(messageR, self.AnswerStepFour)
    
    def AnswerStepFour(self, message):
        '''
        Method that manages the response provided by the user in StepFour
        '''
        TYPES_DATASETS = np.array(["Public", "Sentinel-Hub", "Private", "Upload", "Default", "Back", "Unmerced", "ChangeDetection"])
        try:
            if not self.stepStartCommands(message):
                # correct option?
                if message.text in TYPES_DATASETS:
                    if message.text in TYPES_DATASETS[[0, 2, 6, 7]]: self.Datasets(message)
                    elif message.text == TYPES_DATASETS[1]:       self.SentinelDatasets(message)                   
                    elif message.text == TYPES_DATASETS[3]:       self.UploadDatasets(message)
                    elif message.text == TYPES_DATASETS[5]:       self.StepThree(message)  # back
                    else:  # default
                        self.listExecution[message.chat.id]['Upload'] = Algorithms.FileDatasets[self.listExecution[message.chat.id]['Operation']][self.listExecution[message.chat.id]['DatasetTypeChosen']][self.listExecution[message.chat.id]['singleOrMultipleImage']]['upload']
                        self.listExecution[message.chat.id]['Dataset'] = Algorithms.FileDatasets[self.listExecution[message.chat.id]['Operation']][self.listExecution[message.chat.id]['DatasetTypeChosen']][self.listExecution[message.chat.id]['singleOrMultipleImage']]['dataset']
                        self.StepFive(message)  # next step
                else:
                    self.bot.send_message(message.chat.id, "ERROR: Select a correct option please")
                    self.StepFour(message)
            else: self.StepFour(message, show_help=True)
        except Exception as excep:
            if str(excep) == 'Waiter': self.StepFour(message)
            else:
                print(sys.exc_info())
                self.bot.send_message(message.chat.id, "Error to select datasets")
                self.StepFour(message)
        except:
            print(sys.exc_info())
            self.bot.send_message(message.chat.id, "Error to select datasets")
            self.StepFour(message)

    def StepFive(self, message, show_help=False):
        '''
        Method that asks if you want to modify the default parameters
        '''
        self.listExecution[message.chat.id]['CurrentStep'] = 5
        options = ["Yes", "No", "Back"]
        
        # buttons
        markup = Tools.createButton(options)
        if show_help:
            messageR = self.bot.send_message(message.chat.id, "Please, follow the provided instructions", reply_markup=markup)
        else:
            messageR = self.bot.send_message(message.chat.id, "Do you want change the recommended parameters of the algorithm?", reply_markup=markup)
        self.bot.register_next_step_handler(messageR, self.AnswerStepFive, options)
    
    def AnswerStepFive(self, message, options):
        '''
        Method that manages the response provided by the user in StepFive
        '''
        if not self.stepStartCommands(message):
            # correct option?
            if message.text in options:
                # parameters recommended
                dataset = self.listExecution[message.chat.id]['Dataset']
                operation = self.listExecution[message.chat.id]['Operation']
                typealg = self.listExecution[message.chat.id]['Type']
                algorithm = self.listExecution[message.chat.id]['Algorithm']
                
                # datasets
                if(dataset in Algorithms.FileParametersRecommended and 
                   operation in Algorithms.FileParametersRecommended[dataset] and 
                   typealg in Algorithms.FileParametersRecommended[dataset][operation] and
                   algorithm in Algorithms.FileParametersRecommended[dataset][operation][typealg] and 
                   self.listExecution[message.chat.id]['Parameters']):
                    for k, v in self.listExecution[message.chat.id]['Parameters'].items():
                        if(k in Algorithms.FileParametersRecommended[dataset][operation][typealg][algorithm]):
                            self.listExecution[message.chat.id]['Parameters'][k]['value'] = Algorithms.FileParametersRecommended[dataset][operation][typealg][algorithm][k]
                
                # modify the parameters
                if message.text == options[0]:
                    self.StepSix(message)
                # back
                elif message.text == options[2]:
                    self.StepFour(message)
                else:
                    # execute method
                    self.StepSeven(message) 
            else:
                self.bot.send_message(message.chat.id, "ERROR: Select a correct option please")
                self.StepFive(message)
        else:
            self.StepFive(message, show_help=True)

    def StepSix(self, message):
        '''
        Method to change the parameters
        '''
        try:
            self.listExecution[message.chat.id]['CurrentStep'] = 6
            
            # best parameters
            dataset = self.listExecution[message.chat.id]['Dataset']
            operation = self.listExecution[message.chat.id]['Operation']
            typealg = self.listExecution[message.chat.id]['Type']
            algorithm = self.listExecution[message.chat.id]['Algorithm']
            bestParameters = False
            
            # datasets
            if(dataset in Algorithms.FileParametersRecommended and 
               operation in Algorithms.FileParametersRecommended[dataset] and 
               typealg in Algorithms.FileParametersRecommended[dataset][operation] and
               algorithm in Algorithms.FileParametersRecommended[dataset][operation][typealg]):
                        bestParameters = True
            if self.listExecution[message.chat.id]['Parameters']:
                for k, v in self.listExecution[message.chat.id]['Parameters'].items():
                    # best parameters
                    recommended = ''
                    if(bestParameters):
                        if(k in Algorithms.FileParametersRecommended[dataset][operation][typealg][algorithm]):
                            recommended = Algorithms.FileParametersRecommended[dataset][operation][typealg][algorithm][k]
                    self.listExecution[message.chat.id]['Parameters'][k]['value'] = self.parameterModifier(message, k, v, recommended)
            else:
                self.bot.send_message(message.chat.id, "There are no parameters to change")
            # next step
            self.StepSeven(message)
            
        except Exception as excep:
            if str(excep) == 'Waiter':
                self.StepFive(message)
            else:
                self.bot.send_message(message.chat.id, "Error in modify parameters")
                self.StepSix(message)
        except:
            self.bot.send_message(message.chat.id, "Error in modify parameters")
            self.StepSix(message)
    
    def StepSeven(self, message, show_help=False):
        '''
        Show information
        '''
        self.listExecution[message.chat.id]['CurrentStep'] = 7
        
        parametersInfo = ""
        if 'Parameters' in self.listExecution[message.chat.id] and self.listExecution[message.chat.id]['Parameters'] != None:
            parametersInfo = "\n\t\t\tParameters: "
            for k, v in self.listExecution[message.chat.id]['Parameters'].items():
                parametersInfo += "\n\t\t\t\t\t\t" + str(k) + ": " + str(v['value'])
           
        parametersSentinelInfo = "" 
        if 'ParametersSentinel' in self.listExecution[message.chat.id] and self.listExecution[message.chat.id]['ParametersSentinel'] != None:
            parametersSentinelInfo = "\n\t\t\tParameters Sentinel: "
            for k, v in self.listExecution[message.chat.id]['ParametersSentinel'].items():
                parametersSentinelInfo += "\n\t\t\t\t\t\t" + str(k) + ": " + str(v['value'])
        
        info = "Information: \n\t\t\tDate: {} \n\t\t\tOperation: {} \n\t\t\tType of Algorithm: {} \n\t\t\tAlgorithm: {} \n\t\t\tUpload: {} \n\t\t\tDataset: {} {} {}"
        info = info.format(self.listExecution[message.chat.id]['Time'], self.listExecution[message.chat.id]['Operation'], self.listExecution[message.chat.id]['Type'], self.listExecution[message.chat.id]['Algorithm'],
                           self.listExecution[message.chat.id]['Upload'], self.listExecution[message.chat.id]['Dataset']
                           , parametersInfo, parametersSentinelInfo)
        
        self.bot.send_message(message.chat.id, info)
        
        options = ["Yes", "No", "Back"]
        
        # buttons
        markup = Tools.createButton(options)
        if show_help:
            messageR = self.bot.send_message(message.chat.id, "Please, follow the provided instructions", reply_markup=markup)
        else:
            messageR = self.bot.send_message(message.chat.id, "Do you want run the algorithm?", reply_markup=markup)
        self.bot.register_next_step_handler(messageR, self.AnswerStepSeven, options, info)
    
    def AnswerStepSeven(self, message, options, info):
        '''
        Method that manages the response provided by the user in StepSeven
        '''
        if not self.stepStartCommands(message):
            if message.text in options:  # correct option?
                if message.text == options[0]:  # execute method
                    path = 'src/Consults/' + str(message.chat.id) + '/'
                    Tools.saveLog(info, path)  # save information in log
                    self.StepEight(message)
                elif message.text == options[2]: self.StepFive(message)  # Back
                else: self.StepZero(message)  # start again
            else:
                self.bot.send_message(message.chat.id, "ERROR: Select a correct option please")
                self.StepSeven(message)
        else:
            self.StepSeven(message, show_help=True)
            
    def StepEight(self, message):
        '''
        Execution Method
        '''
        self.listExecution[message.chat.id]['CurrentStep'] = 8
        try:
            if self.queueExec == None or Algorithms.FileConfig["Bot"]['parallelOrQueueExecutionAlg'] == 'parallel':  # modo paralelo
                mainAlgorithm(self.listExecution, message, self.bot)
            else:  # modo cola
                idThread = threading.currentThread().getName()
                self.queueExec.insertElementQueue(idThread=idThread, message=message)
                self.queueExec.dictEvent[idThread].wait()
                self.queueExec.dictEvent[idThread].clear()
                if self.queueExec.dictResultExecution[idThread] != 0:
                    raise Exception('Error in execution')
            self.StepNine(message)  
        except:
            self.bot.send_message(message.chat.id, "Error in the execution: Dataset or parameters are not correct")
            for png in Tools.lsArch_png('src/Consults/' + str(message.chat.id) + '/'):
                try:
                    os.remove(png)
                except:
                    print("FATAL ERROR IN StepEight. remove")
            self.StepZero(message)  

    def StepNine(self, message, show_help=False):
        '''
        Method that asks if you want to send an email with the results
        '''
        self.listExecution[message.chat.id]['CurrentStep'] = 9
        options = ["Yes", "No"]
        
        # buttons
        markup = Tools.createButton(options)
        if show_help:
            messageR = self.bot.send_message(message.chat.id, "Please, follow the provided instructions", reply_markup=markup)
        else:
            messageR = self.bot.send_message(message.chat.id, "Do you want send a email with the result?", reply_markup=markup)
        self.bot.register_next_step_handler(messageR, self.AnswerStepNine, options)

    def AnswerStepNine(self, message, options):
        '''
        Method that manages the response provided by the user in StepNine
        '''
        try:
            if not self.stepStartCommands(message):
                # correct option?
                if message.text in options:
                    # yes
                    if message.text == options[0]:
                        # create folder
                        if not os.path.exists('src/SendEmails'): os.mkdir('src/SendEmails')
                        if not os.path.exists('src/SendEmails/' + str(message.chat.id)): os.mkdir('src/SendEmails/' + str(message.chat.id))
                        # create zip
                        self.listExecution[message.chat.id]['zipElements'] = \
                            Tools.lsArch_png('src/Consults/' + str(message.chat.id) + '/')
                        
                        pathSave = 'src/SendEmails/' + str(message.chat.id) + '/'
                        zipname = str(self.listExecution[message.chat.id]['Time']) + '.zip'
                        Tools.create_zip(self.listExecution[message.chat.id]['zipElements'], pathSave, zipname)
                        
                        # parameters about email
                        password = Algorithms.FileConfig['Email']['password']
                        from_addrs = Algorithms.FileConfig['Email']['from_addrs']
                        subject = Algorithms.FileConfig['Email']['subject']
                        
                        parametersInfo = ""
                        if 'Parameters' in self.listExecution[message.chat.id] and self.listExecution[message.chat.id]['Parameters'] != None:
                            parametersInfo = "\n\tParameters: "
                            for k, v in self.listExecution[message.chat.id]['Parameters'].items():
                                parametersInfo += "\n\t\t" + str(k) + ": " + str(v['value'])
                           
                        parametersSentinelInfo = "" 
                        if 'ParametersSentinel' in self.listExecution[message.chat.id] and self.listExecution[message.chat.id]['ParametersSentinel'] != None:
                            parametersSentinelInfo = "\n\tParameters Sentinel: "
                            for k, v in self.listExecution[message.chat.id]['ParametersSentinel'].items():
                                parametersSentinelInfo += "\n\t\t" + str(k) + ": " + str(v['value'])
                        
                        body = "\nInformation: \n\tDate: {} \n\tOperation: {} \n\tType of Algorithm: {} \n\tAlgorithm: {} \n\tUpload: {} \n\tDataset: {} {} {} \n\tScore: {} \n\n\nTFG Alejandro Redondo Garcia. \nUniversidad Extremadura, Escuela Politécnica."
                        body = body.format(self.listExecution[message.chat.id]['Time'], self.listExecution[message.chat.id]['Operation'], self.listExecution[message.chat.id]['Type'], self.listExecution[message.chat.id]['Algorithm'],
                                   self.listExecution[message.chat.id]['Upload'], self.listExecution[message.chat.id]['Dataset']
                                   , parametersInfo, parametersSentinelInfo, self.listExecution[message.chat.id]['resultPrecision'])
                        
                        # ask receiver
                        self.listExecution[message.chat.id]['Email'] = {}
                        self.bot.send_message(message.chat.id, 'Enter: to_addrs')
                        if not self.waitListener(message):
                            raise Exception('Waiter')
                        if self.listExecution[message.chat.id]['returned'] == "/stop":
                            self.stop(message)
                        else:
                            self.listExecution[message.chat.id]['Email']['to_addrs'] = self.listExecution[message.chat.id]['returned']
                        
                        # send email
                        try:
                            Tools.SendEmail(self.bot, password, from_addrs, self.listExecution[message.chat.id]['Email']['to_addrs'], subject, body, message, pathSave + zipname, zipname)
                        except:
                            self.StepNine(message)
                        info = 'Information: Data {} Send Email to {} with file {}'.format(self.listExecution[message.chat.id]['Time'], self.listExecution[message.chat.id]['Email']['to_addrs'], zipname)
                        Tools.saveLog(info, pathSave)
                    
                    # remove images of consult 
                    try:
                        for png in Tools.lsArch_png('src/Consults/' + str(message.chat.id) + '/'):
                            if os.path.exists(png): os.remove(png)
                    except:
                        print("FATAL ERROR IN AnswerStepNine. remove")

                    self.StepZero(message)  # start again
                else:
                    self.bot.send_message(message.chat.id, "ERROR: Select a correct option please")
                    self.StepNine(message)
            else:
                self.StepNine(message, show_help=True)
                
        except Exception as excep:
            if str(excep) == 'Waiter':
                self.StepNine(message)
            else:
                self.bot.send_message(message.chat.id, "Error to send email")
                self.StepZero(message)  # start again
        except:
            self.bot.send_message(message.chat.id, "Error to send email")
            self.StepZero(message)  # start again

    ################################################################
    ###########       Auxiliary Method    ##########################
    ################################################################
    def waitListener(self, message):
        self.lock.acquire()
        self.userInWait += 1
        self.lock.release()
        
        if self.numThread <= self.userInWait:
            self.bot.send_message(message.chat.id, "The maximum session limit has been reached, please try again later")
            self.lock.acquire()
            self.userInWait -= 1
            self.lock.release()
            return False
        
        self.listExecution[message.chat.id]['returned'] = ""
        self.listExecution[message.chat.id]['timeWaitListener'] = time.time()
        
        while self.listExecution[message.chat.id]['returned'] == "":
            if(time.time() - self.listExecution[message.chat.id]['timeWaitListener'] > 180):  # timeout in seconds
                self.bot.send_message(message.chat.id, "Time out")
                
                self.lock.acquire()
                self.userInWait -= 1
                self.lock.release()
                return False
            
        self.lock.acquire()
        self.userInWait -= 1
        self.lock.release()
        return True
    
    def listener(self, messages):
        for m in messages:
            if m.content_type == 'text':
                if(m.chat.id in self.listExecution):
                    self.listExecution[m.chat.id]['returned'] = m.text
                    
            elif m.content_type in ['document', 'photo']:
                if(m.chat.id in self.listExecution and self.listExecution[m.chat.id]['CurrentStep'] == 4):
                    if m.content_type == 'photo':
                        file = self.bot.get_file(m.photo[-1].file_id)
                        
                        # name file
                        self.bot.send_message(m.chat.id, 'Enter the name of the photo. Example: photo.jpg (maximum 10 characters)')
                        if not self.waitListener(m):
                            raise Exception('Waiter')
                        if self.listExecution[m.chat.id]['returned'] == "/stop":
                            self.stop(m)
                        else:
                            nameFileReturned = self.listExecution[m.chat.id]['returned']
                            name = nameFileReturned[-10:] if len(nameFileReturned) > 10 else nameFileReturned
                        
                    elif m.content_type == 'document':
                        file = self.bot.get_file(m.document.file_id)
                        name = m.document.file_name
    
                    if name.split(".")[-1] in self.listExecution[m.chat.id]['Extension']:
                        fileD = self.bot.download_file(file.file_path)
    
                        dire = 'datasets/Private'
                        if not os.path.exists(dire): os.mkdir(dire)
                        dire += '/' + str(m.chat.id)
                        if not os.path.exists(dire): os.mkdir(dire)
                        
                        with open(dire + '/' + str(name), 'wb') as new_file:
                            new_file.write(fileD)
                        new_file.close()
                        self.listExecution[m.chat.id]['returned'] = name
                    else:
                        self.bot.send_message(m.chat.id, "Send a correct document please")
                else:
                    self.bot.send_message(m.chat.id, "File is not compatible")
    
    def parameterModifier(self, message, k, v, recommended=None):
        correctValue = False
        while(not correctValue):
            if(v['type'] != 'str'):
                strInfo = "Insert a value to {} \n\tType: {} \n\tMax Value: {} \n\tMin Value: {} \n\tActual Value: {} \n\tRecommended: {}".format(k, v['type'], v['maxValue'], v['minValue'], v['value'], recommended)
            else:
                strInfo = "Insert a value to {} \n\tType: {} \n\tList Values: {} \n\tActual Value: {} \n\tRecommended: {}".format(k, v['type'], v['listValues'], v['value'], recommended)
            self.bot.send_message(message.chat.id, strInfo)
            # wait to answer
            if not self.waitListener(message):
                raise Exception('Waiter')
            
            if self.listExecution[message.chat.id]['returned'] == "/stop":
                self.stop(message)
            
            # correct values?
            if(v['type'] == 'float'):
                if(float(v['minValue']) <= float(self.listExecution[message.chat.id]['returned']) <= float(v['maxValue'])):
                    correctValue = True
                else:
                    self.bot.send_message(message.chat.id, 'The value is not within the ranges')
            elif(v['type'] == 'int'):
                if(int(v['minValue']) <= int(self.listExecution[message.chat.id]['returned']) <= int(v['maxValue'])):
                    correctValue = True
                else:
                    self.bot.send_message(message.chat.id, 'The value is not within the ranges')
            elif(v['type'] == 'str'):
                if(k != 'time_interval_init' and k != 'time_interval_end'):
                    if(self.listExecution[message.chat.id]['returned'] in v['listValues']):
                        correctValue = True
                    else:
                        self.bot.send_message(message.chat.id, 'The value is not within the ranges')
                else:
                    try:
                        if(self.listExecution[message.chat.id]['returned'] in v['listValues'] or datetime.datetime.strptime(self.listExecution[message.chat.id]['returned'], "%Y-%m-%d")):
                            if(k == 'time_interval_end'):
                                if(self.listExecution[message.chat.id]['ParametersSentinel']['time_interval_init']['value'] != 
                                   self.listExecution[message.chat.id]['returned'] and self.listExecution[message.chat.id]['ParametersSentinel']['time_interval_init']['value'] < 
                                   self.listExecution[message.chat.id]['returned']):
                                    correctValue = True
                                else:
                                    self.bot.send_message(message.chat.id, 'TThe final date must be greater than the initial date')
                            else:
                                correctValue = True
                    except:
                        self.bot.send_message(message.chat.id, 'The value is not within the ranges or not is a date')
                
            else:
                self.bot.send_message(message.chat.id, 'Type of value not supported')
                 
        # save new value
        return self.listExecution[message.chat.id]['returned']

    ################################################################
    ###########       Datasets Manager   ##########################
    ################################################################

    def Datasets(self, message):
        '''
        Method to handle dataset
        '''
        singleOrMultiple = 'single'
        typeDataset = self.listExecution[message.chat.id]['DatasetTypeChosen']
        if message.text == 'Public' and typeDataset == 'Hyperspectral':  pathdsets = 'datasets/HSI-datasets/'
        elif message.text == 'Public' and typeDataset == 'Multispectral': pathdsets = 'datasets/Multispectral/'
        elif message.text == 'Public' and typeDataset == 'RGB': pathdsets = 'datasets/RGB/'
        elif message.text == 'Private': pathdsets = 'datasets/Private/' + str(message.chat.id) + "/"
        elif message.text in ['Unmerced']:  # para aniadir mas dataset que sean la carpeta entera
            singleOrMultiple = 'multiple'
            self.listExecution[message.chat.id]['Upload'] = 'datasets/' + typeDataset + '/'
            self.listExecution[message.chat.id]['Dataset'] = message.text + '/'
            self.StepFive(message)
            return  # ya no es necesario seguir ejecutando el modulo
        elif message.text in ['ChangeDetection']:  # para aniadir carpeta entera pero seleccionando una carpeta dentro de esa carpeta
            singleOrMultiple = 'multiple'
            pathdsets = 'datasets/' + message.text + '/'
        
        if message.text == 'Public' and typeDataset == 'Hyperspectral':
            if(self.listExecution[message.chat.id]['Operation'] != 'Unmixing'): pathdsets += 'Classification/'
            else: pathdsets += 'Unmixing/'
        
        self.listExecution[message.chat.id]['Upload'] = pathdsets
        if singleOrMultiple == 'multiple':
            lstNameFiles = Tools.lsFolder(pathdsets)
        else:
            lstNameFiles = Tools.lsDataset(pathdsets, self.listExecution[message.chat.id]['Extension'])
        lstNameFiles.append('Back')
        
        if lstNameFiles:
            correctOpt = False
            # buttons
            markup = Tools.createButton(lstNameFiles)
            while(not correctOpt):
                self.bot.send_message(message.chat.id, "Which dataset do you want to use?", reply_markup=markup)
                if not self.waitListener(message): raise Exception('Waiter')
                if self.listExecution[message.chat.id]['returned'] == "/stop": self.stop(message)
                # correct option?
                if self.listExecution[message.chat.id]['returned'] in lstNameFiles:
                    if(self.listExecution[message.chat.id]['returned'] == 'Back'): self.StepFour(message)
                    else:
                        # save dataset
                        if singleOrMultiple == 'multiple':
                            self.listExecution[message.chat.id]['Dataset'] = self.listExecution[message.chat.id]['returned'] + '/'
                        else:
                            self.listExecution[message.chat.id]['Dataset'] = self.listExecution[message.chat.id]['returned']
                        # modify parameters
                        self.StepFive(message)
                    correctOpt = True
                else:
                    self.bot.send_message(message.chat.id, "ERROR: Select a correct option please")
        else:
            self.bot.send_message(message.chat.id, "No files")
            self.StepFour(message)
  
    def UploadDatasets(self, message):
        '''
        Method to upload the dataset
        '''
        pathdsetpriv = 'datasets/Private/'
        self.listExecution[message.chat.id]['Upload'] = pathdsetpriv + str(message.chat.id) + '/'
        self.bot.send_message(message.chat.id, "Send your own Image dataset from your devide with this extensions: " + str(self.listExecution[message.chat.id]['Extension']))
        correct = False
        while not correct:
            if not self.waitListener(message): raise Exception('Waiter')
            
            if self.listExecution[message.chat.id]['returned'] == "/stop": self.stop(message)
            else:
                time.sleep(0.1)  # es necesario para que no retorne false aunque este el fichero en el directorio
                if os.path.exists(pathdsetpriv + str(message.chat.id) + '/' + self.listExecution[message.chat.id]['returned']): correct = True
                else: self.bot.send_message(message.chat.id, "Send a correct dataset please")
        
        self.listExecution[message.chat.id]['Dataset'] = self.listExecution[message.chat.id]['returned']

        sendSecondFile = False
        if(self.listExecution[message.chat.id]['Operation'] != 'Restoration'):
            nameDataset, extension = str(self.listExecution[message.chat.id]['Dataset']).split(".")
            if((self.listExecution[message.chat.id]['Operation'] in ['Classification', 'DimensionalityReduction']) \
                and self.listExecution[message.chat.id]['Type'] == 'Supervised'):
                self.bot.send_message(message.chat.id, "Send the labels of your own dataset from your devide with name " + nameDataset + "_gt" + extension)
                sendSecondFile = True         
            elif(self.listExecution[message.chat.id]['Operation'] == 'Unmixing'):
                self.bot.send_message(message.chat.id, "Send the End of your own dataset from your devide with name " + nameDataset + "_end" + extension)
                sendSecondFile = True
            else: pass  # ALERT
            if(sendSecondFile):
                correct = False
                while not correct:
                    if not self.waitListener(message): raise Exception('Waiter')
                    if self.listExecution[message.chat.id]['returned'] == "/stop": self.stop(message)
                    else:
                        if os.path.exists(pathdsetpriv + str(message.chat.id) + '/' + self.listExecution[message.chat.id]['returned']): correct = True
                        else: self.bot.send_message(message.chat.id, "Send a correct dataset please")
        self.StepFive(message)
    
    def SentinelDatasets(self, message):
        '''
        Method to Sentinel Datasets
        '''
        if not (self.listExecution[message.chat.id]['Type'] == 'Supervised' and self.listExecution[message.chat.id]['Operation'] == 'Unmixing'):
            try:
                rmtree("datasets/Sentinel/" + str(message.chat.id))
            except:
                print("FATAL ERROR IN SentinelDatasets. remove")
            
            if not os.path.exists("datasets/Sentinel/"): os.mkdir("datasets/Sentinel")
            if not os.path.exists("datasets/Sentinel/" + str(message.chat.id)): os.mkdir("datasets/Sentinel/" + str(message.chat.id))
            
            try:
                # ask the parameters
                self.listExecution[message.chat.id]['ParametersSentinel'] = deepcopy(Algorithms.FileParametersSentinel)
                for k, v in self.listExecution[message.chat.id]['ParametersSentinel'].items():
                    # save new value
                    self.listExecution[message.chat.id]['ParametersSentinel'][k]['value'] = self.parameterModifier(message, k, v)
            except Exception as excep:
                if str(excep) == 'Waiter':
                    raise Exception('Waiter')
                else:
                    self.bot.send_message(message.chat.id, 'Error to modify parameters Sentinel')
                    self.SentinelDatasets(message)
            except:
                self.bot.send_message(message.chat.id, 'Error to modify parameters Sentinel')
                self.SentinelDatasets(message)
                
            # download data
            try:
                INSTANCE_ID = Algorithms.FileConfig["SentinelHub"]["API_TOKEN"]  
                if INSTANCE_ID:
                    config = SHConfig()
                    config.instance_id = INSTANCE_ID
                else:
                    config = None
            
                box_select = [float(self.listExecution[message.chat.id]['ParametersSentinel']['upperLeftCornerLongitude']['value']),
                                          float(self.listExecution[message.chat.id]['ParametersSentinel']['upperLeftCornerLatitude']['value']),
                                          float(self.listExecution[message.chat.id]['ParametersSentinel']['lowerRightCornerLongitude']['value']),
                                          float(self.listExecution[message.chat.id]['ParametersSentinel']['lowerRightCornerLatitude']['value'])]
                roi_bbox = BBox(bbox=box_select, crs=CRS.WGS84)
                resx = self.listExecution[message.chat.id]['ParametersSentinel']['ResolutionX_Meters']['value'] + 'm'  # resolution x in meters
                resy = self.listExecution[message.chat.id]['ParametersSentinel']['ResolutionY_Meters']['value'] + 'm'  # resolution y in meters
                time_interval = (self.listExecution[message.chat.id]['ParametersSentinel']['time_interval_init']['value'], self.listExecution[message.chat.id]['ParametersSentinel']['time_interval_end']['value'])
                layer = 'BANDS-' + self.listExecution[message.chat.id]['ParametersSentinel']['satellite']['value']
                layerRGB = 'TRUE-COLOR-' + self.listExecution[message.chat.id]['ParametersSentinel']['satellite']['value']
                maxcc = float(self.listExecution[message.chat.id]['ParametersSentinel']['maximumAllowedCloudCover']['value'])
                
                if(layer == 'BANDS-LANDSAT8-L1C'):
                    dataSource = DataSource.LANDSAT8
                    input_task = L8L1CWCSInput
                elif (layer == 'BANDS-S2-L1C'):
                    dataSource = DataSource.SENTINEL2_L1C
                    input_task = S2L1CWCSInput
                else:
                    print("Error to select satellite")
                
                input_task = input_task(layer=layer, resx=resx, resy=resy, maxcc=maxcc)
                add_sen2cor = AddSen2CorClassificationFeature('SCL', layer='BANDS-S2-L2A', size_x=resx, size_y=resy)
                workflow = LinearWorkflow(input_task, add_sen2cor)
                result = workflow.execute({input_task: {'bbox': roi_bbox, 'time_interval': time_interval}})
                result = result.eopatch()
                
                numberImagesSave = 1 if self.listExecution[message.chat.id]['Operation'] != 'ChangeDetection' else 2
                indexSave = -1
                for i in range(numberImagesSave, 0, -1):
                    timeImage = result['timestamp'][indexSave]
                    imageAllBands = result['data'][layer][indexSave]
                    imageGT = result['mask']['SCL'][indexSave]
                    
                    if(self.listExecution[message.chat.id]['Operation'] == 'Unmixing'):
                        imageAllBandsAux = np.transpose(imageAllBands, axes=(1, 0, 2)) 
                        imageAllBandsAux = np.reshape(imageAllBandsAux, (-1, np.shape(imageAllBands)[2])).T
                        sio.savemat('./datasets/Sentinel/' + str(message.chat.id) + '/SentinelDataset' + str(i) + '.mat', {'Y':imageAllBandsAux, 'nCol':np.shape(imageAllBands)[0], 'nRow':np.shape(imageAllBands)[1]})
                    else:
                        sio.savemat('datasets/Sentinel/' + str(message.chat.id) + '/SentinelDataset' + str(i) + '.mat', {'SentinelDataset':imageAllBands})
                        sio.savemat('datasets/Sentinel/' + str(message.chat.id) + '/SentinelDataset' + str(i) + '_gt.mat', {'SentinelDatasetGT':imageGT})

                    # download rgb image
                    try:
                        wms_true_color_request = WmsRequest(
                            layer=layerRGB,
                            data_source=dataSource,
                            bbox=roi_bbox,
                            time=timeImage,
                            height=imageGT.shape[0],
                            width=imageGT.shape[1],
                            config=config
                        )
                        wms_true_color_img = wms_true_color_request.get_data()[-1]
                        sio.savemat('datasets/Sentinel/' + str(message.chat.id) + '/SentinelDataset' + str(i) + '_rgb.mat', {'RGBSentinelDataset':wms_true_color_img})
                    except:
                        print("FATAL ERROR IN SentinelDatasets. download rgb image")
                    indexSave += 1
                    
                if self.listExecution[message.chat.id]['Operation'] != 'ChangeDetection':
                    self.listExecution[message.chat.id]['Upload'] = 'datasets/Sentinel/' + str(message.chat.id) + '/'
                    self.listExecution[message.chat.id]['Dataset'] = 'SentinelDataset1.mat'
                else:
                    self.listExecution[message.chat.id]['Upload'] = 'datasets/'
                    self.listExecution[message.chat.id]['Dataset'] = 'Sentinel/' + str(message.chat.id) + '/'
                self.StepFive(message)
                        
            except:
                self.bot.send_message(message.chat.id, "ERROR: Downloding data")
                self.StepFour(message)
                    
        else:
            markup = Tools.createButton(['Back'])
            self.bot.send_message(message.chat.id, "The operation is not available for this type of algorithm", reply_markup=markup)
            correctOpt = False
            while(not correctOpt):
                if not self.waitListener(message):
                    raise Exception('Waiter')
                if self.listExecution[message.chat.id]['returned'] == "/stop":
                    self.stop(message)
                elif(self.listExecution[message.chat.id]['returned'] == 'Back'):
                    correctOpt = True
                    self.StepFour(message)
                else:
                    self.bot.send_message(message.chat.id, "ERROR: Select a correct option please")
    
    ################################################################
    ###########      Commands Method      ##########################
    ################################################################
    def stepStartCommands(self, message):
        if message.text == "/stop": self.stop(message)
        elif message.text == "/help": self.helps(message)
        else: return False
        return True

    def stop(self, message):
        self.bot.stop_polling()
        self.bot.reply_to(message, "Stop Bot")

    def helps(self, message):
        switcherHelps = {
            0 : 'Welcome to the bot. Choose: Hyperspectral, Multispectral or RGB',
            1 : 'Choose: Classification, Dimensionality Reduction, Restoration, Unmixing or Change Detection',
            2 : 'Choose between Supervided, Semisupervised or Unsupervised algorithms',
            3 : 'Choose an algorithm',
            4 : 'Choose where are you datasets',
            5 : 'Choose yes if you want modify the parameters or no if you do no want modify the parameters',
            6 : 'Step to modify the parameters of the algorimths',
            7 : 'Information of the execution. Choose yes if you want execution or no if you do no want execution',
            8 : 'Execution process',
            9 : 'Choose yes if you want send Email or no if you do no want send Email'
        }
        
        try: messageHelp = switcherHelps.get(self.listExecution[message.chat.id]['CurrentStep'])
        except: messageHelp = 'Help is not available'
        self.bot.reply_to(message, messageHelp)

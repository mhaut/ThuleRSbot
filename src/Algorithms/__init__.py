'''
Class for read etc files (yaml and json)
'''
# Import
import yaml
import json
from enum import Enum

######################################################################
#############               YAML / JSON                  #############
######################################################################


def readYALM(pathfile):
    with open(pathfile, 'r') as stream:
        try:
            yalmdata = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
                print(exc)
    return yalmdata


def readJSON(pathfile):
    if not ".json" in pathfile:
        print("Error: the configuration file has to be .json")
        
    else:
        with open(pathfile) as data_file:
            try:
                config_bot = json.load(data_file)
            except json.JSONDecodeError as exc:
                print(exc)
        return config_bot


FileParametersDefault = readYALM("etc/ParametersDefault.yaml")
FileDatasets = readYALM("etc/DatasetsDefault.yaml")
FileParametersRecommended = readYALM("etc/ParametersRecommended.yaml")
FileParametersSentinel = readYALM("etc/ParametersSentinel.yaml")
FileConfig = readJSON("etc/config.json")

######################################################################
#############               Enum Class                   #############
######################################################################


class enumTypeAlgorithms(Enum):
    SUPERVISED = 0
    UNSUPERVISED = 1
    SEMISUPERVISED = 2

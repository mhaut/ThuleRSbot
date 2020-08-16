#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import
import sys
from src.TelegramManager import TelegramManager

if __name__ == "__main__":
    '''
    Main class of the project
    Execution: python3.7 Main.py
    '''
    
    if sys.version_info.major != 3:
        print("ERROR, use python3")
    else:
        if(len(sys.argv) == 1):
            TelegramManager()
        else:
            print("ERROR: Parameters are not necessary")
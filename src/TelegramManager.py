#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import
from src.Commands import Commands
from src import Algorithms
from src.QueueExecution import queueExecution
import telebot
import os
import traceback, time


class TelegramManager(object):
    '''
    bot manager class
    '''
    
    def __init__(self):
        configTelegram = Algorithms.FileConfig["Telegram"]
        self.bot = telebot.TeleBot(token=configTelegram["API_TOKEN"], num_threads=int(configTelegram["num_Threads"]))
        self.listExecution = {}
        if not os.path.exists("src/Consults"): os.mkdir("src/Consults")
        
        # modo cola de ejecucion
        if Algorithms.FileConfig["Bot"]['parallelOrQueueExecutionAlg'] == 'queue':
            self.queueExec = queueExecution(bot=self.bot, listExecution=self.listExecution)
        else: self.queueExec = None  # parallel
        
        Commands(self.bot, self.listExecution, self.queueExec)
        self.telegramPolling()
        
    def telegramPolling(self):
        try:
            self.bot.polling(True)
        except:
            traceback_error_string = traceback.format_exc()
            with open("Error.Log", "a") as myfile:
                myfile.write("\r\n\r\n" + time.strftime("%c") + "\r\n<<ERROR polling>>\r\n" + traceback_error_string + "\r\n<<ERROR polling>>")
            self.bot.stop_polling()
            time.sleep(10)
            self.telegramPolling()

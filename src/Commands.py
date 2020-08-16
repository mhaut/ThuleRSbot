#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import
from src.StepsStart import StepsStart
import os


class Commands(object):
    '''
    Command Manager Class
    '''

    def __init__(self, bot, listExecution, queueExec=None):
        self.bot = bot
        self.listExecution = listExecution
        self.queueExec = queueExec
        self.stepsStart = StepsStart(self.bot, self.listExecution, self.queueExec)
    
        @bot.message_handler(commands=['start'])
        def start(message):
            bot.reply_to(message, "Welcome to the Machine Learning bot")
            if not os.path.exists('src/Consults/' + str(message.chat.id)): os.mkdir('src/Consults/' + str(message.chat.id))
            self.stepsStart.StepZero(message)
        
        @bot.message_handler(commands=['help'])
        def helps(message):
            bot.reply_to(message, 'Enter /start to start the bot')
        
        @bot.message_handler(commands=['about'])
        def about(message):
            bot.reply_to(message, "TFG project. This bot is used to treat hyperspectral images with machine learning techniques.")
        

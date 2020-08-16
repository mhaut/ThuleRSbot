from src.MainAlgorithms import mainAlgorithm
import threading, sys
from copy import deepcopy


class queueExecution(object):

    def __init__(self, bot, listExecution):
        self.bot = bot
        self.listQueueExecution = []
        self.dictEvent = {}
        self.dictResultExecution = {}
        self.dictPriority = {}
        self.eventQueueIsNotEmtpy = threading.Event()
        self.lock = threading.Lock()
        self.listExecution = listExecution
        queueThread = threading.Thread(target=self.executionQueue)
        queueThread.start()

    def priority(self, idUser):
        if idUser not in self.dictPriority:
            self.dictPriority[idUser] = 1.0
        elif self.dictPriority[idUser] != 0:
            self.dictPriority[idUser] -= 0.1
        return self.dictPriority[idUser]

    def insertOrder(self, work):
        enc = False
        for i in range(len(self.listQueueExecution)):
            if self.listQueueExecution[i][0] > work[0]:
                enc = True; break
        if enc: self.listQueueExecution = self.listQueueExecution[:i] + [work] + self.listQueueExecution[i:]
        else: self.listQueueExecution = self.listQueueExecution + [work]

    def insertElementQueue(self, idThread, message):
        self.lock.acquire()
        priority = self.priority(message.chat.id)
        if idThread not in self.dictEvent:
            self.dictEvent[idThread] = threading.Event()
        self.dictResultExecution[idThread] = 0
        work = [priority, [str(idThread), message]]
        if len(self.listQueueExecution) == 0:
            self.listQueueExecution = [[priority, [str(idThread), message]]]
        else:
            self.insertOrder(work)
        self.eventQueueIsNotEmtpy.set()
        self.lock.release()

    def executionQueue(self):
        try:
            while True:
                self.eventQueueIsNotEmtpy.wait()
                self.lock.acquire()
                if len(self.listQueueExecution) > 0:
                    priority, [idThread, message] = self.listQueueExecution.pop()
                if len(self.listQueueExecution) == 0:
                    self.eventQueueIsNotEmtpy.clear()
                self.lock.release()
                
                try:
                    mainAlgorithm(self.listExecution, message, self.bot)
                except:
                    self.dictResultExecution[idThread] = 1
                
                self.dictEvent[idThread].set()
        except:
            print(sys.exc_info())
            self.executionQueue()

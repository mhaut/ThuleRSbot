# https://keras.io/optimizers/
from keras import optimizers

class myoptimizer():
    def __init__(self, method='Adam', lr=None, decay=None, momentum=None, nesterov=None, rho=None,
                beta_1=None, beta_2=None, amsgrad=None):
        if method == "SGD":
            lr = 0.01 if lr == None else lr
            decay = 1e-6 if decay == None else decay
            momentum = 0.9  if momentum == None else momentum
            nesterov = True if nesterov == None else nesterov
            kwargs = {'learning_rate':lr, 'decay':decay, 'momentum':momentum, 'nesterov':nesterov}
            self.opt_select = optimizers.SGD(**kwargs)
        elif method == "RMSprop":
            lr = 0.001 if lr == None else lr
            rho = 0.9  if rho == None else rho
            kwargs = {'learning_rate':lr, 'rho':rho}
            self.opt_select = optimizers.RMSprop(**kwargs)
        elif method == "Adagrad":
            self.opt_select = optimizers.Adagrad(learning_rate=0.01 if lr == None else lr)
        elif method == "Adadelta":
            lr = 1.0  if lr == None else lr
            rho = 0.95 if rho == None else rho
            kwargs = {'learning_rate':lr, 'rho':rho}
            self.opt_select = optimizers.Adadelta(**kwargs)
        elif method == "Adam":
            lr = 0.001 if lr == None else lr
            beta_1 = 0.9 if beta_1 == None else beta_1
            beta_2 = 0.999  if beta_2 == None else beta_2
            amsgrad = False if amsgrad == None else amsgrad
            kwargs = {'learning_rate':lr, 'beta_1':beta_1, 'beta_2':beta_2, 'amsgrad':amsgrad}
            self.opt_select = optimizers.Adam(**kwargs)
        elif method in ["Adamax", "Nadam"]:
            lr = 0.002 if lr == None else lr
            beta_1 = 0.9 if beta_1 == None else beta_1
            beta_2 = 0.999  if beta_2 == None else beta_2
            kwargs = {'learning_rate':lr, 'beta_1':beta_1, 'beta_2':beta_2}
            if method == "Adamax": self.opt_select = optimizers.Adamax(**kwargs)
            else: self.opt_select = optimizers.Nadam(**kwargs)
        else:
            print("ERROR OPTIMIZER")
            exit()
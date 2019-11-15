#See: https://arxiv.org/pdf/1506.03099.pdf
from math import *

class tfr_scheduler:
    def __init__(self, initial=1.0, minimum=.1, coeff=1., roc=0.5,
    reset_step=None, mode="constant", itt=0, tfr_max=0.9):
        self.initial=initial
        self.cur_tfr=initial
        self.minimum=minimum
        self.coeff = coeff
        self.roc = roc
        self.reset_step=reset_step
        self.mode=mode
        self.itt = itt
        self.tfr_max=tfr_max

    def getTFR(self):
        return self.cur_tfr
    def getItt(self):
        return self.itt
    def getLen(self):
        return self.reset_step
    def step(self):
        temp = self.coeff * (float(self.itt % self.reset_step)/float(self.reset_step))

        if self.mode == "constant":
            pass
        elif self.mode == "linear":
            self.cur_tfr = max(self.tfr_max * (self.initial - temp), self.minimum)
        elif self.mode == "exp":
            self.cur_tfr = max(self.tfr_max * (float(self.roc)**(temp)), self.minimum)
        elif self.mode == "inv_sigmoid":
            self.cur_tfr = max(self.tfr_max * (self.roc/(self.roc + exp(temp/self.roc))), self.minimum)

        self.itt += 1
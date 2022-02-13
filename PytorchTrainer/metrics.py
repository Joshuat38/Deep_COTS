# -*- coding: utf-8 -*-
"""
@author: Joshua Thompson
@email: joshuat38@gmail.com
@description: Code written for the Kaggle COTS - Great Barrier Reef competiton.
"""

class BaseMetric(object): # 'object' keyword defines this as a base class and can only be used as a base class.
    def __init__(self, accumulate_limit=100, name='Metric'):
        
        self.name = name
        self.accumulate_limit = accumulate_limit
        self.value = None
        self.eval = False
        self.n = 0
        
    def compute(self, y_pred, y_true):
        
        res = self.call(y_pred, y_true)
        if self.value is None:
            self.value = res
            if self.eval == True:
                self.n += 1
        else:
            if self.eval == False:
                self.value = self.value + (res-self.value)/self.accumulate_limit
            else:
                self.value = self.value + (res-self.value)/(self.n)
                self.n += 1
            
    def call(self, y_pred, y_true):
        pass
        
    def result(self):
        return self.value.item()
    
    def reset(self):
        self.value = None
        
    def train(self):
        self.eval = False
    
    def test(self):
        self.eval = True
        self.n = 0
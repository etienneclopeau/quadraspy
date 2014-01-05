# -*- coding: utf-8 -*-
"""
Created on Fri Nov 01 10:12:37 2013

@author: clopeau
"""
from time import time,sleep

class PID():
    def __init__(self,kp,ki,kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.lastTime = time()
        self.lastErr = 0
        self.errSum = 0.
        
    def compute(self, currentValue, consigne):
        
        now = time()
        timeChange = now - self.lastTime
          
        #Compute all the working error variables
        error =  consigne - currentValue
        self.errSum += (error * timeChange)
        dErr = (error - self.lastErr) / timeChange
        print timeChange
#        print error, self.errSum,dErr  
        #Compute PID Output
        Output = self.kp * error + self.ki * self.errSum + self.kd * dErr
          
        #Remember some variables for next time
        self.lastErr = error
        self.lastTime = now
        
        return Output
        
if __name__ == "__main__":
    
    from math import sqrt
    
    def function(x):
        if x > 0: return sqrt(x)
        else: return -sqrt(-x)
    
    t0 = time()
    pid = PID(2,0,0.01)
    v = 0
    while True:
        sleep(0.1)
#        x = (time() - t0) + 1
        v = pid.compute(function(v),4)
        print v,function(v)

  

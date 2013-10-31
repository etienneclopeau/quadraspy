# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:56:55 2013

@author: clopeau
"""




class Motors():
    def __init__(self):
        self.currentPower = 0.
        self.motor1 =19828 #adress motor 1
        self.motor2 = 1983
        self.motor3 = 8983
        self.motor4 = 9832
        
        
    def setMotorsSpeed(p,dp):
        """ 
        p is the total power        
        dp is array 2*2 of real in [-1,1]"""
        
        
    
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:56:55 2013

@author: clopeau
"""
from RPIO import PWM
        
class Motor():
    def __init__(self,PWMservo , pin):
        self.minpulse = 700 #microsecond
        self.maxpulse = 2000 #microsecond
        self.pin = pin
        self.curentSpeed = 0.

        self.PWMservo = PWMservo
        
    def setSpeed(self,speed):
        #speed is between 0 to 1
        pulse = self.minpulse + speed*(self.maxpulse-self.minpulse)
        
        self.PWMservo.set_servo(self.pin , pulse)
    
    def calibrate(self):
        self.setSpeed(self,1)
        print """ calibration of ESC on id """,self.id
        print """maxspeed selected 
               you can now connect the battery
               then wait 2s in press enter"""
        input()
        self.setspeed(0)
        time.sleep(5)
        print """ your ESC should know be calibrated"""
        
        


class Motors():
    def __init__(self):
        self.currentPower = 0.
        self.motor1 = 19828 #adress motor 1
        self.motor2 = 1983
        self.motor3 = 8983
        self.motor4 = 9832
        
        
    def setMotorsSpeed(p,dp):
        """ 
        p is the total power        
        dp is array 2*2 of real in [-1,1]"""
  

    def calibrate(motor = 1):
        if motor = 1: self.motor1.calibrate()
        if motor = 2: self.motor2.calibrate()
        if motor = 3: self.motor3.calibrate()
        if motor = 4: self.motor4.calibrate()
      
if __name__ == "__main__":
      
#PWMservo = PWM.Servo(dma_channel=0, subcycle_time_us=20000, pulse_incr_us=10))
#
## Add servo pulse for GPIO 17 with 1200µs (1.2ms)
#servo.set_servo(17, 1200)
#
## Add servo pulse for GPIO 17 with 2000µs (2.0ms)
#servo.set_servo(17, 2000)
#
## Clear servo on GPIO17
#servo.stop_servo(17)
    
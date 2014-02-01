# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:56:55 2013

@author: clopeau
"""
# import logging
# log_format = '%(levelname)s | %(asctime)-15s | %(message)s'
# logging.basicConfig(format=log_format, level=logging.WARNING)

import time
try :
    from RPIO import PWM
    PWMservo = PWM.Servo(dma_channel=0, subcycle_time_us=20000, pulse_incr_us=1)
except:
    print 'Warning: RPIO not availabe'
    class noRPIO():
        def __init__(self):
            pass
        def set_servo(self):
            pass
    PWMservo = noRPIO()

PWM.set_loglevel(PWM.LOG_LEVEL_ERRORS)

class Motor():
    def __init__(self,PWMservo , pin):
        self.minpulse = 1000 #microsecond
        self.maxpulse = 2000 #microsecond
        self.pin = pin
        self.curentSpeed = 0.

        self.PWMservo = PWMservo
        
    def setSpeed(self,speed):
        #speed is between 0 to 1
        if speed > 1. : speed =1.
        elif speed < 0. : speed = 0.
        pulse = int(self.minpulse + speed*(self.maxpulse-self.minpulse))
        
        self.PWMservo.set_servo(self.pin , pulse)
    
    def calibrate(self):
        print """ calibration of ESC on id """,self.pin
        self.setSpeed(1)
        print """maxspeed selected 
               you can now connect the battery
               then wait 2s in press enter"""
        raw_input()
        self.setSpeed(0)
        time.sleep(5)
        print """ your ESC should know be calibrated"""
 
    def test(self):
        print """ test of ESC on id """,self.pin
        self.setSpeed(0)
        print """NE PAS INSTALLER LES HELICES !!!
        please connect the battery and press enter"""
        raw_input()
    
        v = 0
        while v < 1:
            v += 0.01
            self.setSpeed(v)
            time.sleep(0.5)
        time.sleep(2)
        while v > 0:
            v -= 0.01
            self.setSpeed(v)
            time.sleep(0.1)
        print """ motor test done"""
    
        


class Motors():
    def __init__(self):
        self.currentPower = 0.
        self.motor1 = Motor(PWMservo,17)     #adress motor 1
        self.motor2 = Motor(PWMservo,18)
        self.motor3 = Motor(PWMservo,22)
        self.motor4 = Motor(PWMservo,23)
        
        self.motor1.setSpeed(0.)
        self.motor2.setSpeed(0.)
        self.motor3.setSpeed(0.)
        self.motor4.setSpeed(0.)
        
    def setMotorsSpeed(self, p):
        """ 
        p is array 2*2 of real in [0,1]"""
        self.motor1.setSpeed(p[0,0])
        self.motor2.setSpeed(p[0,1])
        self.motor3.setSpeed(p[1,0])
        self.motor4.setSpeed(p[1,1])
  

    def calibrate(self, motor = 'all'):
        if motor == 'all' : 
            print """ calibration of all ESC """
            self.motor1.setSpeed(1)
            self.motor2.setSpeed(1)
            self.motor3.setSpeed(1)
            self.motor4.setSpeed(1)
            print """maxspeed selected 
                   you can now connect the battery
                   then wait 2s in press enter"""
            raw_input()
            self.motor1.setSpeed(0)
            self.motor2.setSpeed(0)
            self.motor3.setSpeed(0)
            self.motor4.setSpeed(0)
            time.sleep(5)
            print """ your ESC should now be calibrated"""

        elif motor == 1: self.motor1.calibrate()
        elif motor == 2: self.motor2.calibrate()
        elif motor == 3: self.motor3.calibrate()
        elif motor == 4: self.motor4.calibrate()
    
    def test(self, motor = 1):
        if motor == 1: self.motor1.test()
        if motor == 2: self.motor2.test()
        if motor == 3: self.motor3.test()
        if motor == 4: self.motor4.test()
        
if __name__ == "__main__":
      pass
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

    


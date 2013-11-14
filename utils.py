# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:33:50 2013

@author: clopeau
"""
import sys

from capteurs import getCapteurs
from motors import Motors
from imu import logIMU,plotIMU,timeIMU

import sys
try:
    choice1 = sys.argv[1]
    choice2 = sys.argv[2]
    stopafter = True
except :
    choice1 = None
    choice2 = None
    stopafter = False

#sudo python -m cProfile -s time utils.py 1 perfoIMU > _resProfile


def utils():
    
    while True:
        print """-----------------------------------
        1 --> utils IMU
        2 --> motors
        
        """
        if choice1 == None:
            choice = raw_input('please make a choice and press enter\n')
        else: choice = choice1
        
        if choice == '1':
            utilsIMU()
        elif choice == '2':
            utilsMotors()
        else:
            print "This choice is not reconized, try again"
	
        if stopafter: return

    
    

def utilsIMU():
    
    acc, mag, gyr = getCapteurs()
    while True:
        print '----------------------------------- \nutils for IMU'
        print """-----------------------------------
            utils for IMU
            logAcc --> logging Accelerometers values
            addlogAcc --> logging additional Accelerometers values
            plotAcc --> plot logged accelerometers values
            calAcc --> calibrate accelerometers
    
            logMag --> logging Accelerometers values
            addlogMag --> logging additional Accelerometers values
            plotMag --> plot logged accelerometers values
            calMag --> calibrate accelerometers
            
            calGyr --> calibrate gyrometers
 
            logIMU --> logging IMU data
            plotIMU --> plot logged IMU data
            perfoIMU --> times 1000 iter
         
            0 --> go Back
            
            """
        
        if choice2 == None:
            choice = raw_input('please set action and press enter\n')
        else: choice = choice2
    
        if choice == 'logAcc':
            print 'logging Accelerometers values'
            acc.logValues(runningtime = 60, sleep = 0.1)
        
        elif choice == 'addlogAcc':
            print 'logging additional Accelerometers values'
            acc.logValues(runningtime = 60, sleep = 0.1, add = True)
    
        elif choice == 'plotAcc':
            print 'ploting Accelerometers values'
            acc.plotLogValues()
        
        elif choice == 'calAcc':
            print 'calibrating accelerometers'
            acc.calibrate()
        
        
        
        elif choice == 'logMag':
            print 'logging Magnetometes values'
            mag.logValues( runningtime = 60, sleep = 0.1)
        elif choice == 'addlogMag':
            print 'logging additional Magnetometes values'
            mag.logValues(runningtime = 60, sleep = 0.1, add = True)
        
        elif choice == 'plotMag':
            print 'ploting Magnetometers values'
            mag.plotLogValues()
        
        elif choice == 'calMag':
            print 'calibrating magnetometers'
            mag.calibrate()
        
        
        
        elif choice == 'calGyr':
            print 'calibrating gyrometers'
            gyr.calibrate()
        
        elif choice == 'logIMU':
            logIMU()
                
        elif choice == 'plotIMU':
            plotIMU()
            
        elif choice == 'perfoIMU':
            timeIMU(niter = 1000)
            
        elif choice == '0':
            return
    
        else:
            print "This choice is not reconized, try again"
	
	if stopafter: return



def utilsMotors():
    motors = Motors()
    while True:
        print "---------------------------"
        print "utils for Motors"
        print "---------------------------"
        print """ 
              1 --> calibrate ESC
	      2 --> test motor
              
              0 --> go back
              """
        if choice2 == None:
            choice = raw_input('please set action and press enter\n')
        else: choice = choice2
    
        if choice == '1':
            choice = raw_input('please set the motor id to ccalibrate\n')
            motors.calibrate(int(choice))

        elif choice == '2':
            choice = raw_input('please set the motor id to test\n')
            motors.test(int(choice))

        elif choice == '0':
            return
            
        else:
            print "This choice is not reconized, try again"


if __name__ == "__main__":
    utils()            


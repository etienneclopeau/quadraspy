# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:33:50 2013

@author: clopeau
"""
import sys

from capteurs import getCapteurs

acc, mag, gyr = getCapteurs()

argv = sys.argv

if '--logAcc' in argv:
    print 'logging Accelerometers values'
    acc.logValues(runningtime = 60, sleep = 0.1)
if '--addlogAcc' in argv:
    print 'logging additional Accelerometers values'
    acc.logValues(runningtime = 60, sleep = 0.1, add = True)

if '--plotAcc' in argv:
    print 'ploting Accelerometers values'
    acc.plotLogValues()

if '--calAcc' in argv:
    print 'calibrating accelerometers'
    acc.calibrate()



if '--logMag' in argv:
    print 'logging Magnetometes values'
    mag.logValues( runningtime = 60, sleep = 0.1)
if '--addlogMag' in argv:
    print 'logging additional Magnetometes values'
    mag.logValues(runningtime = 60, sleep = 0.1, add = True)

if '--plotMag' in argv:
    print 'ploting Magnetometers values'
    mag.plotLogValues()

if '--calMag' in argv:
    print 'calibrating magnetometers'
    mag.calibrate()



if '--calGyr' in argv:
    print 'calibrating gyrometers'
    gyr.calibrate()


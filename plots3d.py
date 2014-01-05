from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import pyqtgraph.opengl as gl
import time
from numpy.linalg import norm as npnorm

from imu import IMU

option = 'acc'  # 'mag' or 'acc'


import Pyro4
# import time
# uri=raw_input("What is the Pyro uri of the greeting object? ").strip()

# -- Initialize Pyro client
import Pyro4.naming, Pyro4.core, sys
#Pyro4.core.initClient()

#-- Locate the Name Server
print 'Searching Name Server...'

# Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
Pyro4.config.SERIALIZER = 'pickle'

ns = Pyro4.locateNS(host = '192.168.1.98', port = 9090)
print 'Name Server found '

imu=Pyro4.Proxy(ns.lookup("r_IMU"))          # get a Pyro proxy to the greeting object


app = QtGui.QApplication([])
view = gl.GLViewWidget()

# view.opts['distance'] = 20.
view.setCameraPosition(distance = 20)
view.show()
view.setWindowTitle('magnetometers')

g = gl.GLGridItem()
view.addItem(g)

pos3 = np.zeros((20000,3))
sp3 = gl.GLScatterPlotItem(pos=pos3, color=(1,1,1,.3), size=0.1, pxMode=False)

view.addItem(sp3)


# 
i = 0
if option == 'mag':
    flog = open('log/logMagForCalib.dat','w')
elif option == 'acc':
    flog = open('log/logAccForCalib.dat','w')
imu.stop()
def update():
    global pos3,i,flog
    imu.getMeasurements()
    accx,accy,accz,magx,magy,magz,gyrx,gyry,gyrz = imu.getRawData()
    psi,theta,phi = imu.getEuler()
    # dataTime.append(time.time()-t0)
    # mag/=npnorm(mag)
    if option == 'mag':
        pos3[i,0]=magx/500.
        pos3[i,1]=magy/500.
        pos3[i,2]=magz/500.
        # pos3[i,0]=1.
        # pos3[i,1]=0.
        # pos3[i,2]=0.
        i+=1
        print i,magx,magy,magz
        sp3.setData(pos=pos3)
        flog.write('%s %s %s\n'%(magx,magy,magz))
    elif option == 'acc':
        if i == 0:
            pos3[i,0]=accx
            pos3[i,1]=accy
            pos3[i,2]=accz
        else:
            pos3[i,0]= 0.9 * pos3[i-1,0] + 0.1 * accx
            pos3[i,1]= 0.9 * pos3[i-1,1] + 0.1 * accy
            pos3[i,2]= 0.9 * pos3[i-1,2] + 0.1 * accz

        # pos3[i,0]=1.
        # pos3[i,1]=0.
        # pos3[i,2]=0.
    
        print i,accx,accy,accz
        sp3.setData(pos=pos3)
        flog.write('%s %s %s\n'%(pos3[i,0],pos3[i,1],pos3[i,2]))
        i+=1

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)




QtGui.QApplication.instance().exec_()



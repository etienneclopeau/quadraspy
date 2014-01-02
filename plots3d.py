from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import pyqtgraph.opengl as gl
import time
from numpy.linalg import norm as npnorm

from imu import IMU


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

pos3 = np.zeros((10000,3))
sp3 = gl.GLScatterPlotItem(pos=pos3, color=(1,1,1,.3), size=0.1, pxMode=False)

view.addItem(sp3)


# 
i = 0
flog = open('log/logMagForCalib.dat','w')
imu.stop()
def update():
    global pos3,i,flog
    tcurrent,acc,mag,gyr = imu.getMeasurements()
    psi,theta,phi = imu.getEuler()
    # dataTime.append(time.time()-t0)
    # mag/=npnorm(mag)
    pos3[i,0]=mag[0]/500.
    pos3[i,1]=mag[1]/500.
    pos3[i,2]=mag[2]/500.
    # pos3[i,0]=1.
    # pos3[i,1]=0.
    # pos3[i,2]=0.
    i+=1
    print i,mag
    sp3.setData(pos=pos3)
    flog.write('%s %s %s\n'%(mag[0],mag[1],mag[2]))

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)




QtGui.QApplication.instance().exec_()



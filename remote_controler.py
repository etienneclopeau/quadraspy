from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import pyqtgraph.opengl as gl
import time

from imu import IMU


import Pyro4
import time
#uri=raw_input("What is the Pyro uri of the greeting object? ").strip()

#-- Initialize Pyro client
import Pyro4.naming, Pyro4.core, sys
#Pyro4.core.initClient()

#-- Locate the Name Server
print 'Searching Name Server...'
Pyro4.config.SERIALIZER = 'pickle'
ns = Pyro4.locateNS(host = '192.168.1.98', port = 9090)
print 'Name Server found '


imu=Pyro4.Proxy(ns.lookup("r_IMU"))          # get a Pyro proxy to the greeting object
altimeter = Pyro4.Proxy(ns.lookup("r_ALTIMETER"))
motors = Pyro4.Proxy(ns.lookup("r_MOTORS"))
quad = Pyro4.Proxy(ns.lookup("r_QUAD"))

class animatedPlot():
    def __init__(self,lay,title,lines):
        self.maxtime = 20.
        self.mintime = 0.
        self.deltat = 10.


        self.p = lay.addPlot(title = title)
        self.p.addLegend()
        self.p.showGrid(y=True)
        self.lines = list()
        self.data = list()
        self.data.append(list())  #time
        for line in lines:
            self.data.append(list())
            self.lines.append(self.p.plot([0],[0],pen='y',name = line))

    def append(self,values):
        for i,(d, v) in  enumerate(zip(self.data, values)):
            d.append(v)
        for l,d in zip(self.lines, self.data[1:]):
            l.setData(self.data[0],d)
        if self.maxtime < self.data[0][-1]:
            self.mintime += self.deltat
            self.maxtime += self.deltat
            self.p.setRange(xRange = (self.mintime,self.maxtime))


app = QtGui.QApplication([])
view = pg.GraphicsView()
lay = pg.GraphicsLayout(border=(100,100,100))
view.setCentralItem(lay)
view.show()
view.setWindowTitle('IMU plots')
view.resize(800,600)

mag = animatedPlot(lay,'magnetometers',('magx','magy','magz'))
euler = animatedPlot(lay,'Euler',('psi','theta','phi','alt'))
motors = animatedPlot(lay,'motors',('mot1','mot2','mot3','mot4'))

lay.nextRow()
acc = animatedPlot(lay,'accelerometers',('accx','accy','accz'))
Emag = animatedPlot(lay,'earth magnetic field',('x','z'))

lay.nextRow()
gyr = animatedPlot(lay,'gyrometres',('gyrx','gyry','gyrz'))
gyrErr = animatedPlot(lay,'gyrometres biais',('gyrEx','gyrEy','gyrEz'))



#imu = IMU(log = False, logSleep = 0., simu = 'log/_imuLog_2013Dec08_13h28m06s')

t0 = time.time()
def update():
    global imu, dataTime,t0
    global dataAccx,dataAccy,dataAccz
    global dataMagx,dataMagy,dataMagz
    global dataGyrx,dataGyry,dataGyrz
    global magCx,magCy,magCz
    global accCx,accCy,accCz
    global gyrCx,gyrCy,gyrCz
    global gyrbCx,gyrbCy,gyrbCz
    global dataEmagx,dataEmagz
    global magP,gyrP,accP,maxtime,mintime
    accx,accy,accz,magx,magy,magz,gyrx,gyry,gyrz = imu.getRawData()
    psi,theta,phi = imu.getEuler()
    gyr_ba = imu.get_eInt()
    emagx,emagz = imu.getEarth_mag()
    alt = altimeter.getAltitude()
    T1 = time.time()-t0
    mag.append((T1,magx,magy,magz))
    acc.append((T1,accx,accy,accz))
    gyr.append((T1,np.degrees(gyrx),np.degrees(gyry),np.degrees(gyrz)))
    gyrErr.append((T1,gyr_ba[0],gyr_ba[1],gyr_ba[2]))
    euler.appendr((T1,psi,theta,phi,alt))
    Emag.append((T1,emagx,emagz))
    motors.append((T1,m1,m2,m3,m4))


    app.processEvents()  ## force complete redraw for every plot

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)




QtGui.QApplication.instance().exec_()



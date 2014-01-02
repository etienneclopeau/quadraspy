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




app = QtGui.QApplication([])
view = pg.GraphicsView()
lay = pg.GraphicsLayout(border=(100,100,100))
view.setCentralItem(lay)
view.show()
view.setWindowTitle('IMU plots')
view.resize(800,600)

magP = lay.addPlot(title = "magnetometers")
# magP.setRange(xRange = (0,10),yRange = (-1,1))
magP.showGrid(y=True)
magCx = magP.plot([0],[0],pen='y')
magCy = magP.plot([0],[0],pen='b')
magCz = magP.plot([0],[0],pen='r')
# magWidget = gl.GLViewWidget()
# lay.addWidget(magWidget)

EulerP = lay.addPlot(title = "Euler")
EulerP.showGrid(y=True)
psiC = EulerP.plot([0],[0],pen='y')
thetaC = EulerP.plot([0],[0],pen='b')
phiC = EulerP.plot([0],[0],pen='r')


lay.nextRow()
accP = lay.addPlot(title = "accelerometres")
accP.showGrid(y=True)
accCx = accP.plot([0],[0],pen='y')
accCy = accP.plot([0],[0],pen='b')
accCz = accP.plot([0],[0],pen='r')

EmagP = lay.addPlot(title = "earth magnetic field")
EmagP.showGrid(y=True)
EmagCx = EmagP.plot([0],[0],pen='y')
EmagCz = EmagP.plot([0],[0],pen='r')


lay.nextRow()
gyrP = lay.addPlot(title = "gyrometres")
gyrP.showGrid(y=True)
gyrCx = gyrP.plot([0],[0],pen='y')
gyrCy = gyrP.plot([0],[0],pen='b')
gyrCz = gyrP.plot([0],[0],pen='r')

gyrbP = lay.addPlot(title = "gyrometres biais")
gyrbP.showGrid(y=True)
gyrbCx = gyrbP.plot([0],[0],pen='y')
gyrbCy = gyrbP.plot([0],[0],pen='b')
gyrbCz = gyrbP.plot([0],[0],pen='r')




#imu = IMU(log = False, logSleep = 0., simu = 'log/_imuLog_2013Dec08_13h28m06s')
dataTime = list()
dataAccx = list()
dataAccy = list()
dataAccz = list()
dataMagx = list()
dataMagy = list()
dataMagz = list()
dataGyrx = list()
dataGyry = list()
dataGyrz = list()
dataGyrbx = list()
dataGyrby = list()
dataGyrbz = list()
dataTheta = list()
dataPsi = list()
dataPhi = list()
dataEmagx = list()
dataEmagz = list()
maxtime = 20.
mintime = 0.
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
    dataTime.append(time.time()-t0)
    dataAccx.append(accx)
    dataAccy.append(accy)
    dataAccz.append(accz)
    dataMagx.append(magx)
    dataMagy.append(magy)
    dataMagz.append(magz)
    dataGyrx.append(gyrx)
    dataGyry.append(gyry)
    dataGyrz.append(gyrz)
    dataGyrbx.append(gyr_ba[0])
    dataGyrby.append(gyr_ba[1])
    dataGyrbz.append(gyr_ba[2])
    dataPsi.append(psi)
    dataTheta.append(theta)
    dataPhi.append(phi)
    dataEmagx.append(emagx)
    dataEmagz.append(emagz)

    magCx.setData(dataTime,dataMagx)
    magCy.setData(dataTime,dataMagy)
    magCz.setData(dataTime,dataMagz)

    accCx.setData(dataTime,dataAccx)
    accCy.setData(dataTime,dataAccy)
    accCz.setData(dataTime,dataAccz)

    gyrCx.setData(dataTime,dataGyrx)
    gyrCy.setData(dataTime,dataGyry)
    gyrCz.setData(dataTime,dataGyrz)

    gyrbCx.setData(dataTime,dataGyrbx)
    gyrbCy.setData(dataTime,dataGyrby)
    gyrbCz.setData(dataTime,dataGyrbz)

    psiC.setData(dataTime,dataPsi)
    thetaC.setData(dataTime,dataTheta)
    phiC.setData(dataTime,dataPhi)

    EmagCx.setData(dataTime,dataEmagx)
    EmagCz.setData(dataTime,dataEmagz)

    #time range
    if maxtime < dataTime[-1]:
        mintime += 10.
        maxtime += 10.
        magP.setRange(xRange = (mintime,maxtime))
        accP.setRange(xRange = (mintime,maxtime))
        gyrP.setRange(xRange = (mintime,maxtime))
        EulerP.setRange(xRange = (mintime,maxtime))
        gyrbP.setRange(xRange = (mintime,maxtime))
        EmagP.setRange(xRange = (mintime,maxtime))

    app.processEvents()  ## force complete redraw for every plot

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)




QtGui.QApplication.instance().exec_()



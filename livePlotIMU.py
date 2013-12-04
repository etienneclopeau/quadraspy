# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 14:19:42 2013

@author: clopeau
"""


SHOW = True # Show test in GUI-based test launcher

import os.path as osp

from guidata.qt.QtGui import QMainWindow, QWidget, QGridLayout
from guidata.qt.QtCore import QTimer

from guiqwt.image import ImagePlot
from guiqwt.curve import PlotItemList
from guiqwt.histogram import ContrastAdjustment
from guiqwt.plot import PlotManager, CurveWidget
from guiqwt.builder import make
import threading
from math import sqrt

from numpy import array

class CentralWidget(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent)
    
        layout = QGridLayout()
        self.setLayout(layout)
        
        self.plot11 = CurveWidget(self,title = "AccX",xlabel = "Time")
        layout.addWidget(self.plot11, 0, 0, 1, 1)
        self.curve11 = make.curve([1,2,30], [0,3,10], color='b')
        self.plot11.plot.add_item(self.curve11)
        self.plot11.plot.set_axis_limits('left',-1.5,1.5)  
 

        self.plot12 = CurveWidget(self,title = "AccY",xlabel = "Time")
        layout.addWidget(self.plot12, 0, 1, 1, 1)
        self.curve12 = make.curve([1,2,30], [0,3,10], color='b')
        self.plot12.plot.add_item(self.curve12)
        self.plot12.plot.set_axis_limits('left',-1.5,1.5)  

        self.plot13 = CurveWidget(self,title = "AccZ",xlabel = "Time")
        layout.addWidget(self.plot13, 0, 2, 1, 1)
        self.curve13 = make.curve([1,2,30], [0,3,10], color='b')
        self.plot13.plot.add_item(self.curve13)
        self.plot13.plot.set_axis_limits('left',-1.5,1.5)  

        
        self.plot21 = CurveWidget(self,title = "GyrX",xlabel = "Time")
        layout.addWidget(self.plot21, 1, 0, 1, 1)
        self.curve21 = make.curve([1,2,3], [2,3,1], color='b')
        self.plot21.plot.add_item(self.curve21)
        self.plot21.plot.set_axis_limits('left',-90.,90.)  

        self.plot22 = CurveWidget(self,title = "GyrY",xlabel = "Time")
        layout.addWidget(self.plot22, 1, 1, 1, 1)
        self.curve22 = make.curve([1,2,3], [2,3,1], color='b')
        self.plot22.plot.add_item(self.curve22)
        self.plot22.plot.set_axis_limits('left',-90.,90.)

        self.plot23 = CurveWidget(self,title = "GyrZ",xlabel = "Time")
        layout.addWidget(self.plot23, 1, 2, 1, 1)
        self.curve23 = make.curve([1,2,3], [2,3,1], color='b')
        self.plot23.plot.add_item(self.curve23)
        self.plot23.plot.set_axis_limits('left',-90.,90.) 
      
      
        self.plot31 = CurveWidget(self,title = "H_X",xlabel = "Time")
        layout.addWidget(self.plot31, 2, 0, 1, 1)
        self.curve31 = make.curve([1,2,3], [2,3,1], color='b')
        self.plot31.plot.add_item(self.curve31)
        self.plot31.plot.set_axis_limits('left',-3.,3.)

        self.plot32 = CurveWidget(self,title = "H_Y",xlabel = "Time")
        layout.addWidget(self.plot32, 2, 1, 1, 1)
        self.curve32 = make.curve([1,2,3], [2,3,1], color='b')
        self.plot32.plot.add_item(self.curve32)
        self.plot32.plot.set_axis_limits('left',-3.,3.)

        self.plot33 = CurveWidget(self,title = "H_Z",xlabel = "Time")
        layout.addWidget(self.plot33, 2, 2, 1, 1)
        self.curve33 = make.curve([1,2,3], [2,3,1], color='b')
        self.plot33.plot.add_item(self.curve33)
        self.plot33.plot.set_axis_limits('left',-3.,3.)


        self.plot41 = CurveWidget(self,title = "phi",xlabel = "Time")
        layout.addWidget(self.plot41, 3, 0, 1, 1)
        self.curve41 = make.curve([1,2,3], [2,3,1], color='b')
        self.plot41.plot.add_item(self.curve41)
        self.plot41.plot.set_axis_limits('left',-3.,3.)

        self.plot42 = CurveWidget(self,title = "theta",xlabel = "Time")
        layout.addWidget(self.plot42, 3, 1, 1, 1)
        self.curve42 = make.curve([1,2,3], [2,3,1], color='b')
        self.plot42.plot.add_item(self.curve42)
        self.plot42.plot.set_axis_limits('left',-3.,3.)

        self.plot43 = CurveWidget(self,title = "psi",xlabel = "Time")
        layout.addWidget(self.plot43, 3, 2, 1, 1)
        self.curve43 = make.curve([1,2,3], [2,3,1], color='b')
        self.plot43.plot.add_item(self.curve43)
        self.plot43.plot.set_axis_limits('left',-3.,3.)


        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(2)        


    def update(self):
        import time
        from capteurs import getCapteurs
        from imu import IMU
    
        acc, mag, gyr = getCapteurs()
        imu = IMU()
    
        Ttime = list()
        Taccx = list()
        Taccy = list()
        Taccz = list()
        Thx = list()
        Thy = list()
        Thz = list()
        Twx = list()
        Twy = list()
        Twz = list()
        Tphi,Ttheta,Tpsi = list(),list(),list()
        i = 0
        t0 = time.time()
        while True:
            Ttime.append(time.time()-t0) 
            i+=1
            print i
#            Ttime.append(i)
            accx,accy,accz = acc.getAcc()
#            accx,accy,accz = sqrt(i),sqrt(i/2),1
            Taccx.append(accx)
            Taccy.append(accy)
            Taccz.append(accz)
            hx,hy,hz = mag.getMag()
#            hx,hy,hz = 1,1,1
            Thx.append(hx)
            Thy.append(hy)
            Thz.append(hz)
            wx,wy,wz = gyr.getGyr()
#            wx,wy,wz = 1,1,1
            Twx.append(wx)
            Twy.append(wy)
            Twz.append(wz)
            
            psi,theta,phi = imu.update([accx,accy,accz],[hx,hy,hz],[wx,wy,wz])
            Tphi.append(phi)
            Ttheta.append(theta)
            Tpsi.append(psi)

            mintime = min(Ttime)
            maxtime = max(Ttime)            
            
            self.curve11.set_data(Ttime, Taccx)
            self.plot11.plot.set_axis_limits('bottom',mintime,maxtime)  
            self.curve11.plot().replot()
            
            self.curve12.set_data(Ttime, Taccy)
            self.plot12.plot.set_axis_limits('bottom',mintime,maxtime)  
            self.curve12.plot().replot()
            
            self.curve13.set_data(Ttime, Taccz)
            self.plot13.plot.set_axis_limits('bottom',mintime,maxtime)  
            self.curve13.plot().replot()


            self.curve21.set_data(Ttime, Twx)
            self.plot21.plot.set_axis_limits('bottom',mintime,maxtime)  
            self.curve21.plot().replot()
            
            self.curve22.set_data(Ttime, Twy)
            self.plot22.plot.set_axis_limits('bottom',mintime,maxtime)  
            self.curve22.plot().replot()
            
            self.curve23.set_data(Ttime, Twz)
            self.plot23.plot.set_axis_limits('bottom',mintime,maxtime)  
            self.curve23.plot().replot()


            self.curve31.set_data(Ttime, Thx)
            self.plot31.plot.set_axis_limits('bottom',mintime,maxtime)  
            self.curve31.plot().replot()
            
            self.curve32.set_data(Ttime, Thy)
            self.plot32.plot.set_axis_limits('bottom',mintime,maxtime)  
            self.curve32.plot().replot()
            
            self.curve33.set_data(Ttime, Thz)
            self.plot33.plot.set_axis_limits('bottom',mintime,maxtime)  
            self.curve33.plot().replot()


            self.curve41.set_data(Ttime, Thx)
            self.plot41.plot.set_axis_limits('bottom',mintime,maxtime)  
            self.curve41.plot().replot()
            
            self.curve42.set_data(Ttime, Thy)
            self.plot42.plot.set_axis_limits('bottom',mintime,maxtime)  
            self.curve42.plot().replot()
            
            self.curve43.set_data(Ttime, Thz)
            self.plot43.plot.set_axis_limits('bottom',mintime,maxtime)  
            self.curve43.plot().replot()

            #print "updating"            
            #time.sleep(0.5)
            
                     

class Window(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        
        self.widget = CentralWidget(self)
        self.setCentralWidget(self.widget)
#        updating = threading.Thread(target = self.widget.update)
#        updating.start()
        
        
        
def test():
    """Test"""
    # -- Create QApplication
    import guidata
    app = guidata.qapplication()
    # --    
    win = Window()
    win.show()
    app.exec_()


if __name__ == "__main__":
    test()
        

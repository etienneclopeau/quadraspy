# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:53:33 2013

@author: clopeau
"""
from numpy import cos, sin, array


from adxl345 import adxl345
from itg3205 import itg3205
from hmc5883l import hmc5883l
from srf02 import Srf02
from calibration import getData, ellipsoidFit_DistanceSphere,logValues, plot,getCalData

fileCalAcc = '_calibrationAcc.dat'
fileLogAcc = '_logAcc.dat'
fileCalMag = '_calibrationMag.dat'
fileLogMag = '_logMag.dat'
fileCalGyr = '_calibrationGyr.dat'


class Acc(adxl345):
    def __init__(self):
        adxl345.__init__(self)
        try :
            self.Mat,self.center,self.r = getCalData(fileCalAcc)
        except : 
            print 'WARNING : no calibration data evailable for accelerometers'
            self.Mat = array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
            self.center = array([0.,0.,0.])
            self.r = array([1.,1.,1.])
        
            
    def logValues(self, runningtime = 30, sleep = 0.1, add = False):
        logValues(self, fileLogAcc, runningtime , sleep, add )

    def plotLogValues( self ):
        Tv = getData( fileLogAcc)
        plot(Tv)
    
                
    def calibrate(self):
        Th = getData(fileLogAcc)
        call = ellipsoidFit_DistanceSphere(Th)
        f = open(fileCalAcc,'w')
        f.write('%s %s %s %s %s %s %s %s %s \n'%(call[0],call[1],call[2],call[3],call[4],call[5],call[6],call[7],call[8]))
        f.close()
        self.Mat,self.center,self.r = getCalData(fileCalAcc)
        
    def getAcc(self):
        return self.Mat.dot(self.getRawAxes()-self.center)/self.r 


#######################################################################"
class Mag(hmc5883l):
    def __init__(self):
        hmc5883l.__init__(self)
        #self.setmode(self.HMC5883L_MODE_CONTINUOUS)
        #self.setDataRate(self.HMC5883L_RATE_75)
        self.setSampleAveraging(self.HMC5883L_AVERAGING_8)
        try :
            self.Mat,self.center,self.r = getCalData(fileCalMag)
        except : 
            print 'WARNING : no calibration data evailable for magnetometers'
            self.Mat = array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
            self.center = array([0.,0.,0.])
            self.r = array([1.,1.,1.])
        
            
    def logValues(self, runningtime = 30, sleep = 0.1, add = False):
        logValues(self, fileLogMag, runningtime , sleep, add )

    def plotLogValues( self ):
        Tv = getData( fileLogMag)
        plot(Tv)
                
    def calibrate(self):
        Th = getData(fileLogMag)
        call = ellipsoidFit_DistanceSphere(Th)
        f = open(fileCalMag,'w')
        f.write('%s %s %s %s %s %s %s %s %s \n'%(call[0],call[1],call[2],call[3],call[4],call[5],call[6],call[7],call[8]))
        f.close()
        self.Mat,self.center,self.r = getCalData(fileCalMag)
        
    def getMag(self):
        return self.Mat.dot(self.getRawAxes()-self.center)/self.r 


class Gyr(itg3205):
    def __init__(self):
        itg3205.__init__(self)
        try :
            f = open(fileCalGyr)
            line = f.readline()
            f.close()
            (self.dx,self.dy,self.dz) = (float(a) for a in line.split())            
        except : 
            print 'WARNING : no calibration data available for gyrometers'
            self.dx,self.dy,self.dz = 0,0,0
    
    def getGyr(self):
        wx,wy,wz = self.getRadPerSecAxes()
        return array([wx-self.dx, wy-self.dy, wz-self.dz])
#        return wx, wy, wz

    def calibrate(self):
        import time
        self.dx,self.dy,self.dz = 0,0,0  #on enleve la calibration si elle existait
        dx,dy,dz = 0,0,0
        t0 = time.time()
        i = 0
        while time.time()-t0 < 60:
            i += 1
            print i
            wx,wy,wz = self.getGyr()
            dx += wx
            dy += wy
            dz += wz
        self.dx = dx/i
        self.dy = dy/i
        self.dz = dz/i
        f = open(fileCalGyr,'w')
        f.write('%s %s %s\n'%(self.dx,self.dy,self.dz))
        f.close()
        print 'biais gyro = ',self.dx,self.dy,self.dz
            

class Altimeter():
    def __init__(self):
        # self.ultrason = Srf02()

    def getAltitude(slef):
        # dist = self.ultrason.getValue()
        dist = 1.0
        return dist


    
def getCapteurs():
    acc = Acc()
    acc.setScale(4)
    
    mag = Mag()
    
    gyr = Gyr()
    
    return acc, mag, gyr

if __name__ == "__main__":
    import time
    acc,mag,gyr = getCapteurs()
    
    while True:
        print 'acc ',acc.getAcc()
        print 'mag ',mag.getMag()
        print 'gyr ',gyr.getGyr()
        print'--------------------------------------'
        time.sleep(1)
    

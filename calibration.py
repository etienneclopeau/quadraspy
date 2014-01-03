# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:57:18 2013

@author: clopeau
"""
import time
from numpy import array, sqrt, cos,sin,arcsin,arctan2,vstack,ones,eye
from numpy.linalg import norm,inv,eig

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize as so



#class calibrater():
#    
def getData( filename):
    f = open(filename)
    Th = list()
    nt = 0
    for line in f :
        nt += 1
        magno_x, magno_y, magno_z =line.split()
        Th.append( [float(magno_x), float(magno_y), float(magno_z)] )
        #if float(magno_x) == 200.0: print 'x200'
    Th = array(Th)
    return Th
#        
#        
#    def costFunction(self, (matcoef11,matcoef22,matcoef33,offset1,offset2,offset3)):
#        self.cost = 0.
#        #matCoef= array([[matcoef11,0.,0.],
#        #                [matcoef21,matcoef22,0.],
#        #                [matcoef31,matcoef32,matcoef33]])
#        matCoef= array([[matcoef11,0.,0.],
#                        [0,matcoef22,0.],
#                        [0,0,matcoef33]])
#        offset = array([offset1,offset2,offset3])
#        for i in range(self.nt):
#            self.cost += abs(norm(getCalibratedValues(matCoef, offset , self.Th[i,:]))-1.)
#	return self.cost
#
#    def printCurrentState(self, (matcoef11,matcoef22,matcoef33,offset1,offset2,offset3)):
#        #matCoef= array([[matcoef11,0.,0.],
#        #                [matcoef21,matcoef22,0.],
#        #                [matcoef31,matcoef32,matcoef33]])
#        matCoef= array([[matcoef11,0.,0.],
#                        [0,matcoef22,0.],
#                        [0,0,matcoef33]])
#        offset = array([offset1,offset2,offset3])
#        print matCoef
#        print offset
#        print self.cost        
#
#    def calibrate(self):
#        
#        matcoef11 = 0.002958
#        matcoef12 = 0.
#        matcoef13 = 0.
#        matcoef21 = 0.
#        matcoef22 = 0.002731
#        matcoef23 = 0.
#        matcoef31 = 0.
#        matcoef32 = 0.
#        matcoef33 = 0.002862
#        
#        offset1 = 138.198061
#        offset2 = -74.854358
#        offset3 = -6.908972
#        
#        res = so.fmin(self.costFunction,
#                (matcoef11,matcoef22,matcoef33,offset1,offset2,offset3),
#                callback = self.printCurrentState)   
#        return res
#    
#    def costFunctionE(self,(A11,A12,A13,A21,A22,A23,A31,A32,A33)):
#        A = array([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]])     
#        c= array([(max(self.Th[:,0])+min(self.Th[:,0]))/2.,(max(self.Th[:,1])+min(self.Th[:,1]))/2.,(max(self.Th[:,2])+min(self.Th[:,2]))/2.])
#        self.cost = 0
#        for il in range(self.nt):
#            x = array([self.Th[il,0], self.Th[il,1], self.Th[il,2]])
#            self.cost += ((x-c).transpose().dot(A).dot(x-c)-1)**2
#        return self.cost
#    
#    def printCurrentStateE(self,(A11,A12,A13,A21,A22,A23,A31,A32,A33)):
#        A = array([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]])     
#        
#        print A
#        print self.cost
#    
#    def fitElipsoide(self):
#        cx,cy,cz = (max(self.Th[:,0])+min(self.Th[:,0]))/2.,  \
#                   (max(self.Th[:,1])+min(self.Th[:,1]))/2.,  \
#                   (max(self.Th[:,2])+min(self.Th[:,2]))/2.
#        A11,A12,A13,A21,A22,A23,A31,A32,A33= (1./max(self.Th[:,0]-cx)-1./min(self.Th[:,0]-cx))/2. ,0,0,  \
#                                              0,(1./max(self.Th[:,1]-cy)-1./min(self.Th[:,1]-cy))/2.,0,  \
#                                              0,0,(1./max(self.Th[:,2]-cz)-1./min(self.Th[:,2]-cz))/2.
#        res = so.fmin(self.costFunctionE,
#        #res = so.fmin_bfgs(self.costFunctionE,
#                (A11,A12,A13,A21,A22,A23,A31,A32,A33),
#                callback = self.printCurrentStateE)   
#        return res
            
    
def plot( Th, Thc= None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', aspect='equal')
    ax.scatter(Th[:,0], Th[:,1], Th[:,2])
    if not Thc == None: ax.scatter(Thc[:,0], Thc[:,1], Thc[:,2],c='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    fig2 = plt.figure()
    ax = fig2.add_subplot(411)
    ax.plot(Th[:,0])
    if not Thc == None: ax.plot(Thc[:,0])
    ax = fig2.add_subplot(412)
    ax.plot(Th[:,1])
    if not Thc == None: ax.plot(Thc[:,1])
    ax = fig2.add_subplot(413)
    ax.plot(Th[:,2])
    if not Thc == None: ax.plot(Thc[:,2])
    ax = fig2.add_subplot(414)
    ax.plot(sqrt(Th[:,0]**2+ Th[:,1]**2+ Th[:,2]**2))
    if not Thc == None: ax.plot(sqrt(Thc[:,0]**2+ Thc[:,1]**2+ Thc[:,2]**2))
    plt.show()
    
def ellipsoidFitAlgebraic(Th):
    
    # fit ellipsoid in the form Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz = 1
    x,y,z = Th[:,0],Th[:,1],Th[:,2]
    D = vstack( (x * x, 
               y * y, 
               z * z, 
              2 * x * y, 
              2 * x * z, 
              2 * y * z, 
              2 * x, 
              2 * y,  
              2 * z )).transpose() #  ndatapoints x 9 ellipsoid parameters
    v = inv( D.transpose().dot(D) ).dot(  D.transpose().dot( ones( (len(x),1) )  ))[:,0]

    # form the algebraic form of the ellipsoid
    A = array([ [v[0], v[3], v[4], v[6]], \
                [v[3], v[1], v[5], v[7]], \
                [v[4], v[5], v[2], v[8]], \
                [v[6], v[7], v[8], -1  ] ])
    # find the center of the ellipsoid
    center = -inv(A[ 0:3, 0:3 ]).dot( [ v[6], v[7], v[8] ])
    # form the corresponding translation matrix
    T = eye( 4 )
    T[ 3, 0:3 ] = center.transpose()
    # translate to the center
    R = T.dot(A).dot( T.transpose())
    # solve the eigenproblem
    evals ,evecs = eig( R[ 0:3, 0:3 ] / -R[ 3, 3 ] )

    radii = sqrt( 1. / evals )
    print 'center',center
    print 'radii',radii
    print 'evecs',evecs
    print 'v',v
    
    return center,radii,evecs,v

def calibratedData(Th, center, evecs):
    Thc = array([evecs.dot(v-center) for v in Th])
    return Thc
        
def distanceToEllipsoid(a,b,c,u,v,w):
    
    #print '========'

    def Ft(t,u,v,w,a,b,c):
        """fonction to minimize to find the projection of point (u,v,w) 
        on Ellipsoide defined by x**2/a**2 + y**2/b**2 + z**2/c**2"""
        F = a**2*u**2/(t+a**2)**2 + b**2*v**2/(t+b**2)**2 + c**2*w**2/(t+c**2)**2 - 1
        return F
        
    def dFt(t,u,v,w,a,b,c):
        """derivative of Ft"""
        dF = -2*a**2*u**2/(t+a**2)**3 - 2*b**2*v**2/(t+b**2)**3 - 2*c**2*w**2/(t+c**2)**3
        return dF

    def ddFt(t,u,v,w,a,b,c):
        """derivative of Ft"""
        ddF = 6*a**2*u**2/(t+a**2)**4 + 6*b**2*v**2/(t+b**2)**4 + 6*c**2*w**2/(t+c**2)**4
        return ddF

    #print a,b,c,u,v,w

    flipu=False
    flipv=False
    flipw=False
    if u<0: 
        u= -u
        flipu=True
    if v<0: 
        v= -v
        flipv=True
    if w<0: 
        w= -w
        flipw=True


    x0 = max(a*u-a**2,b*v-b**2,c*w-c**2)
    #(res,) = so.fmin_bfgs(Ft, x0, fprime=dFt, args=(u,v,w,a,b,c))
    res = so.newton(Ft, x0, fprime=dFt, fprime2=ddFt, args=(u,v,w,a,b,c), tol=0.001)
    #res = so.fsolve(Ft, x0, fprime=dFt, args=(u,v,w,a,b,c))
    #res = so.anderson(Ft, x0, fprime=dFt, args=(u,v,w,a,b,c))
    #res = so.root(Ft, x0, args=(u,v,w,a,b,c))
    #res = res.x[0]
    #print res
    x = a**2*u/(res+a**2)
    y = b**2*v/(res+b**2)
    z = c**2*w/(res+c**2)
    #print x,y,z
    #print x,y,z
    #print norm([x-u,y-v,z-w])
    return norm([x-u,y-v,z-w])
                
def ellipsoidFit_DistanceSphere(Th):
    center, radii, evecs, v = ellipsoidFitAlgebraic(Th)
    a0,b0,c0 = radii[0],radii[1],radii[2]
    cx0,cy0,cz0 = center[0],center[1],center[2]
    theta0 = -arcsin(evecs[2,0])
    psi0 = arctan2(evecs[2,1]/cos(theta0),evecs[2,2]/cos(theta0))
    phi0 = arctan2(evecs[1,0]/cos(theta0),evecs[0,0]/cos(theta0))
    theta0 = 0
    psi0 = 0
    phi0 = 0
    
    
    def costFunction(x): 
        (theta,psi,phi,a,b,c,cx,cy,cz) = x
        cost = 0.
        #transformation en fonction des parametres de l'ellipse pour se ramener a une sphere:
        Mat = array(  \
          [[cos(theta)*cos(phi) , sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi) , cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)],  \
           [cos(theta)*sin(phi) , sin(psi)*sin(theta)*sin(phi) + cos(psi)*cos(phi) , cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi)],  \
           [   -sin(theta)      ,               sin(psi)*cos(theta)                , cos(psi)*cos(theta)] ] )
        center = array([cx,cy,cz])
        r = array([a,b,c])
        
        Thcal = array([Mat.transpose().dot(v-center)/r for v in Th])
        # si les mesures ne sont pas bruitees et la calibration pafaite, tous les points doivent maintenant être sur une sphere de rayon 1        
        
        for point in Thcal:
            cost += abs(norm(point)-1.)
            #cost += (norm(point)-1.)**2
        print cost,theta,psi,phi,a,b,c,cx,cy,cz
        return cost
    
    x0 = (theta0,psi0,phi0,a0,b0,c0,cx0,cy0,cz0)
    print 'X0',x0
    cost0 = costFunction(x0)
    res = so.fmin_bfgs(costFunction, x0)
    print 'X0 was ' , x0
    print 'cost0',cost0
    print 'res',res
    return res

def ellipsoidFit_DistanceEllipsoide(Th):
    center, radii, evecs, v = ellipsoidFitAlgebraic(Th)
    a0,b0,c0 = radii[0],radii[1],radii[2]
    cx0,cy0,cz0 = center[0],center[1],center[2]
    theta0 = -arcsin(evecs[2,0])
    psi0 = arctan2(evecs[2,1]/cos(theta0),evecs[2,2]/cos(theta0))
    phi0 = arctan2(evecs[1,0]/cos(theta0),evecs[0,0]/cos(theta0))
    phi0,theta0,psi0 = 0,0,0
    print 'test'
    
    
    def costFunction(x): 
        (theta,psi,phi,a,b,c,cx,cy,cz) = x
        cost = 0.
        #transformation en fonction des parametres de l'ellipse pour se ramene a une sphere:
        Mat = array(  \
          [[cos(theta)*cos(phi) , sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi) , cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)],  \
           [cos(theta)*sin(phi) , sin(psi)*sin(theta)*sin(phi) + cos(psi)*cos(phi) , cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi)],  \
           [   -sin(theta)      ,               sin(psi)*cos(theta)                , cos(psi)*cos(theta)] ] )
        center = array([cx,cy,cz])
        r = array([a,b,c])
        
        Thcal = array([Mat.transpose().dot(v-center) for v in Th])
        # si les mesures ne sont pas bruitees et la calibration pafaite, tous les points doivent maintenant être sur une ellipse dont les axes sont aligné sur les axes x,y,z        
        costs = list()
        for point in Thcal:
            costs.append(distanceToEllipsoid(a,b,c,point[0],point[1],point[2]))
            #if costs[-1]>1000: raise
            cost += costs[-1]
        print cost,theta,psi,phi,a,b,c,cx,cy,cz

        # fig2 = plt.figure()
        # ax = fig2.add_subplot(111)
        # ax.plot(costs)
        # plt.show()


        return cost
    
    x0 = (theta0,psi0,phi0,a0,b0,c0,cx0,cy0,cz0)
    print 'X0',x0
    cost0 = costFunction(x0)
    res = so.fmin_bfgs(costFunction, x0, disp=True)
    #res = so.fmin(costFunction, x0)
    print 'X0 was ' , x0
    print 'cost0',cost0
    print 'res',res
    return res
            
        
def logValues(capteur, filename,runningtime = 30, sleep = 0.1, add = False):
    """ this function read the magnetometers values during 'runningtime' second. 
    you have to turn the magnetometer in all direction.
    It is better to make that when all pieces are monted"""
    from math import sqrt
    if add == True: f = open(filename,'a')
    else: f = open(filename,'w')
    i = 0
    t0 = time.time()
    while time.time()-t0 < runningtime:
        i+=1
        print i
        x,y,z = capteur.getRawAxes()
        f.write('%s %s %s\n'%(x,y,z))
        print x,y,z 
        if sleep :
            time.sleep(sleep)  
    f.close()        

def getCalData(fileName):
    f = open(fileName)
    line = f.readline()
    f.close()
    cal = [float(a) for a in line.split()]
    theta,psi,phi,a,b,c,cx,cy,cz = cal
    Mat = array(  \
      [[cos(theta)*cos(phi) , sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi) , cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)],  \
       [cos(theta)*sin(phi) , sin(psi)*sin(theta)*sin(phi) + cos(psi)*cos(phi) , cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi)],  \
       [   -sin(theta)      ,               sin(psi)*cos(theta)                , cos(psi)*cos(theta)] ] )
    center = array([cx,cy,cz])
    r = array([a,b,c])
    return Mat,center,r



import numpy

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y



if __name__ == "__main__":     
    # from pylab import *
    from numpy import zeros
    #logHvalues('maglog.dat',runningtime = 30)   
    print 'read data' 
    # Th = getData('log/logMagForCalib.dat')
    Th = getData('log/logAccForCalib.dat')
    plot(Th[:,0])
    plot(Th[:,1])
    plot(Th[:,2])
    nval = Th.shape[0]
    wlen = 5
    Thf = zeros((nval+wlen-1,3))
    Thf[:,0] = smooth(Th[:,0],wlen)
    Thf[:,1] = smooth(Th[:,1],wlen)
    Thf[:,2] = smooth(Th[:,2],wlen)
    # plot(Thf[:,0])
    # plot(Thf[:,1])
    # plot(Thf[:,2])
    # show()
    #plot(Th)
    print 'lancement opti'
    # ellipsoidFit_DistanceSphere(Thf)
    ellipsoidFit_DistanceEllipsoide(Thf)
    
  #   theta,psi,phi,a,b,c,cx,cy,cz = [1.16461960e-01 , -1.22692113e-01 ,  5.51200083e-01 ,  5.52765112e+02,
  #  5.21074288e+02 ,  4.48637709e+02 ,  1.45520786e+02 , -5.25322073e+01,
  # -1.40086242e+01]

  #   theta,psi,phi,a,b,c,cx,cy,cz = [-1.16681928e-01 ,  1.22478698e-01 , -2.59043829e+00 ,  5.52752085e+02,
  #  5.21075812e+02 ,  4.48639923e+02 ,  1.45509558e+02 , -5.25339048e+01,
  # -1.40101198e+01]
  #   theta,psi,phi,a,b,c,cx,cy,cz = [ 0.0162383499677, -0.11500901018, 3.28073261891 ,487.423874818, 442.976679127 ,414.588916629, 132.442428135 ,-96.8858269441, -24.5031275703 ]

  #   Mat = array(  \
  #     [[cos(theta)*cos(phi) , sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi) , cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)],  \
  #      [cos(theta)*sin(phi) , sin(psi)*sin(theta)*sin(phi) + cos(psi)*cos(phi) , cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi)],  \
  #      [   -sin(theta)      ,               sin(psi)*cos(theta)                , cos(psi)*cos(theta)] ] )
  #   center = array([cx,cy,cz])
  #   r = array([a,b,c])
  #   Thcal = array([Mat.transpose().dot(v-center) for v in Th])
  #   plot(Th,Thcal)

    


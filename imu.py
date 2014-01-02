# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:07:31 2013

@author: clopeau
"""
import threading
from time import gmtime, strftime, time, sleep
from numpy import   array,  sqrt,zeros,cross,arctan,arcsin,radians,degrees,arctan2
from numpy.linalg import norm as npnorm

 
def conj(q):
    return array([q[0],-q[1],-q[2],-q[3]])



class IMU():
    """class IMU
    based on http:#www.x-io.co.uk/res/doc/madgwick_internal_report.pdf
    hearth frame:
        z: vertical vers le haut
        x: horizontal aligned with magnetic field
        y; complete
    euler definition and order
    psi: rotation around z
    theta: rotatioin around y
    phi: rotation around x
    """
    
    def __init__(self, log = True, logSleep = 0., simu = False, start = True, algo = 3):
        self.tbefore = time()

        self.log_ = log
        self.logSleep = logSleep
        if self.log_ :
            logFileName = strftime("log/_imuLog_%Y%b%d_%Hh%Mm%Ss", gmtime())
            self.logFile = open(logFileName,'w')
        

        if simu == False:
            from capteurs import getCapteurs
            self.getMeasurements = self.getMeasurements_real
            self.accelerometer, self.magnetometer, self.gyrometer = getCapteurs()
        else:
            self.simuFile = open(simu)
            self.getMeasurements = self.getMeasurements_simu
        
        self.getMeasurements()



        self.kp = 2
        self.ki = 0.005
        self.beta = 0.1

        if algo == 0 :
            self.update = UpdateMadgwickAHRS
        elif algo == 1 :
            self.update = UpdateMadgwickIMU
        elif algo == 3 :
            self.update = UpdateMahonyAHRS
        elif algo == 4 :
            self.update = UpdateMahonyIMU


        self.quat = [1.,0.,0.,0.]
        self.earth_magnetic_field_x = 1. # orientation of earth magnetic field in ground coordinates
        self.earth_magnetic_field_z = 0. 
        self.eInt = [0.,0.,0.]
        self.deltat = 0.
        self.running = True


        if start:
            self.start()

    def start(self):
        self.threads = list()
        if self.log_ and self.logSleep > 0:
            self.threads.append(threading.Thread(target = self.run))
            self.threads.append(threading.Thread(target = self.runLog))
        elif self.log_ and self.logSleep == 0:
            self.threads.append(threading.Thread(target = self.runAndrunLog))
        else:
            self.threads.append(threading.Thread(target = self.run))
        for t in self.threads:
            t.start()
    


    def run(self):
        while self.running:
            self.getMeasurements()
            self.update()
    def runLog(self):
        while self.running:
            self.log()
            time.sleep(self.logSleep)
    def runAndrunLog(self):
        while self.running:
            self.getMeasurements()
            self.update()
            self.log()


    def stop(self):
        self.running = False

    def getMeasurements_real(self):
        """
        update the measurement data and time data from sensors
        """
        self.tbefore = self.tcurrent
        self.tcurrent = time()
        self.deltat = self.tcurrent - self.tbefore
        self.acc = self.accelerometer.getAcc()
        self.gyr = self.gyrometer.getGyr()
        self.mag = self.magnetometer.getMag()

    def getMeasurements_simu(self):
        """
        update the measurement data and time data from a logFile
        this is used for simulations
        """
        try:
            self.tbefore = self.tcurrent
            logValues = self.simuFile.readline().split()
            res = [float(v) for v in logValues[:10]]
            # sleep(0.1)
            print res[0]
            self.tcurrent = res[0]
            self.acc = array(res[1:4])
            self.mag = array(res[4:7])
            self.gyr = array(res[7:10])
            self.deltat = self.tcurrent - self.tbefore
        except: 
            print "endOfLog"
            self.stop()
        
    def getRawData(self):
        """
        return the actuel measured data
        this is used for remote plotting
        """
        accx,accy,accz = self.acc[0],self.acc[1],self.acc[2]
        magx,magy,magz = self.mag[0],self.mag[1],self.mag[2]
        gyrx,gyry,gyrz = self.gyr[0],self.gyr[1],self.gyr[2]
        return accx,accy,accz, \
               magx,magy,magz, \
               gyrx,gyry,gyrz
    def get_eInt(self):
        """
        return the actuel gyro biais
        this is used for remote plotting
        """
        return self.eInt

    def getEarth_mag(self):
        """
        return the actuel estimated earth magnetic field
        this is used for remote plotting
        """
        return self.earth_magnetic_field_x,self.earth_magnetic_field_z

    def getEuler(self):
        quat0 =  self.quat[0]
        quat1 = -self.quat[1]
        quat2 = -self.quat[2]
        quat3 = -self.quat[3]

#        phi = arctan(2.*(self.quat0*self.quat1+self.quat2*self.quat3)/(1-2*(self.quat1**2+self.quat2**2)))
#        theta = arcsin(2*(self.quat0*self.quat2-self.quat3*self.quat1))
#        psi = arctan(2.*(self.quat0*self.quat3+self.quat1*self.quat2)/(1-2*(self.quat2**2+self.quat3**2)))

        psi = arctan2(2.*(quat1*quat2-quat0*quat3), 2.*(quat0**2+quat1**2)-1.)
        theta = -arcsin(2.*(quat1*quat3+quat0*quat2))
        phi = arctan2(2.*(quat2*quat3-quat0*quat1) , 2.*(quat0**2+quat3**2)-1.)

        # Compute the Euler angles from the quaternion.
        # phi = arctan2(2 * ASq_3 * ASq_4 - 2 * ASq_1 * ASq_2, 2 * ASq_1 * ASq_1 + 2 * ASq_4 * ASq_4 - 1);
        # theta = arcsin(2 * ASq_2 * ASq_3 - 2 * ASq_1 * ASq_3);
        # psi = arctan2(2 * ASq_2 * ASq_3 - 2 * ASq_1 * ASq_4, 2 * ASq_1 * ASq_1 + 2 * ASq_2 * ASq_2 - 1);
     

        #print phi,theta,psi
        return psi, theta, phi
    
    def log(self):    
        psi,theta,phi = self.getEuler()          
        self.logFile.write(('%s '*23+'\n')%(self.tcurrent,self.deltat,
                                              self.acc[0],self.acc[1],self.acc[2],
                                              self.mag[0],self.mag[1],self.mag[2],
                                              self.gyr[0],self.gyr[1],self.gyr[2],
                                              self.eInt[0],self.eInt[1],self.eInt[2], 
                                              self.earth_magnetic_field_x,self.earth_magnetic_field_z ,   
                                              self.quat[0],self.quat[1],self.quat[2],self.quat[3],
                                              psi,theta,phi))
        
        
    """

    Implementation of Madgwick's IMU and AHRS algorithms.

    This Implementation is the python equivalent os this:
    http://www.x-io.co.uk/node/8#open_source_ahrs_and_imu_algorithms
    """


    def UpdateMadgwickAHRS(self):
        """ 
        madgwickAHRS from https://github.com/xioTechnologies/Open-Source-AHRS-With-x-IMU
        Algorithm AHRS update method. Requires gyroscope accelerometer and magnetometer data.
         "gx" Gyroscope x axis measurement in radians/s.
         "gy" Gyroscope y axis measurement in radians/s.
         "gz" Gyroscope z axis measurement in radians/s.
         "ax" Accelerometer x axis measurement in any calibrated units.
         "ay" Accelerometer y axis measurement in any calibrated units.
         "az" Accelerometer z axis measurement in any calibrated units.
         "mx" Magnetometer x axis measurement in any calibrated units.
         "my" Magnetometer y axis measurement in any calibrated units.
         "mz" Magnetometer z axis measurement in any calibrated units.
         
         paremeters:
          self.beta
          self.deltat

         input/output: self.quaternion
        """ 
        # short name local variable for readability
        q1 = self.quat[0]
        q2 = self.quat[1]
        q3 = self.quat[2]
        q4 = self.quat[3]

        ax = self.acc[0]
        ay = self.acc[1]
        az = self.acc[2]
        gx = self.gyr[0]
        gy = self.gyr[1]
        gz = self.gyr[2]
        mx = self.mag[0]
        my = self.mag[1]
        mz = self.mag[2]

        # Auxiliary variables to avoid repeated arithmetic
        _2q1 = 2. * q1
        _2q2 = 2. * q2
        _2q3 = 2. * q3
        _2q4 = 2. * q4
        _2q1q3 = 2. * q1 * q3
        _2q3q4 = 2. * q3 * q4
        q1q1 = q1 * q1
        q1q2 = q1 * q2
        q1q3 = q1 * q3
        q1q4 = q1 * q4
        q2q2 = q2 * q2
        q2q3 = q2 * q3
        q2q4 = q2 * q4
        q3q3 = q3 * q3
        q3q4 = q3 * q4
        q4q4 = q4 * q4

        # Normalise accelerometer measurement
        norm = sqrt(ax * ax + ay * ay + az * az)
        if (norm == 0.) : raise
        norm = 1. / norm        
        ax *= norm
        ay *= norm
        az *= norm

        # Normalise magnetometer measurement
        norm = sqrt(mx * mx + my * my + mz * mz)
        if (norm == 0.) : raise 
        norm = 1. / norm        
        mx *= norm
        my *= norm
        mz *= norm

        # Reference direction of Earth's magnetic field
        _2q1mx = 2. * q1 * mx
        _2q1my = 2. * q1 * my
        _2q1mz = 2. * q1 * mz
        _2q2mx = 2. * q2 * mx
        hx = mx * q1q1 - _2q1my * q4 + _2q1mz * q3 + mx * q2q2 + _2q2 * my * q3 + _2q2 * mz * q4 - mx * q3q3 - mx * q4q4
        hy = _2q1mx * q4 + my * q1q1 - _2q1mz * q2 + _2q2mx * q3 - my * q2q2 + my * q3q3 + _2q3 * mz * q4 - my * q4q4
        _2bx = sqrt(hx * hx + hy * hy)
        _2bz = -_2q1mx * q3 + _2q1my * q2 + mz * q1q1 + _2q2mx * q4 - mz * q2q2 + _2q3 * my * q4 - mz * q3q3 + mz * q4q4
        _4bx = 2. * _2bx
        _4bz = 2. * _2bz

        # Gradient decent algorithm corrective step
        s1 = -_2q3 * (2. * q2q4 - _2q1q3 - ax) + _2q2 * (2. * q1q2 + _2q3q4 - ay) - _2bz * q3 * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (-_2bx * q4 + _2bz * q2) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + _2bx * q3 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)
        s2 = _2q4 * (2. * q2q4 - _2q1q3 - ax) + _2q1 * (2. * q1q2 + _2q3q4 - ay) - 4. * q2 * (1. - 2. * q2q2 - 2. * q3q3 - az) + _2bz * q4 * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (_2bx * q3 + _2bz * q1) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + (_2bx * q4 - _4bz * q2) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)
        s3 = -_2q1 * (2. * q2q4 - _2q1q3 - ax) + _2q4 * (2. * q1q2 + _2q3q4 - ay) - 4. * q3 * (1. - 2. * q2q2 - 2. * q3q3 - az) + (-_4bx * q3 - _2bz * q1) * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (_2bx * q2 + _2bz * q4) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + (_2bx * q1 - _4bz * q3) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)
        s4 = _2q2 * (2. * q2q4 - _2q1q3 - ax) + _2q3 * (2. * q1q2 + _2q3q4 - ay) + (-_4bx * q4 + _2bz * q2) * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (-_2bx * q1 + _2bz * q3) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + _2bx * q2 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)
        norm = 1. / sqrt(s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4)    # normalise step magnitude
        s1 *= norm
        s2 *= norm
        s3 *= norm
        s4 *= norm

        # Compute rate of change of quaternion
        qDot1 = 0.5 * (-q2 * gx - q3 * gy - q4 * gz) - self.beta * s1
        qDot2 = 0.5 * (q1 * gx + q3 * gz - q4 * gy) - self.beta * s2
        qDot3 = 0.5 * (q1 * gy - q2 * gz + q4 * gx) - self.beta * s3
        qDot4 = 0.5 * (q1 * gz + q2 * gy - q3 * gx) - self.beta * s4

        # Integrate to yield quaternion
        q1 += qDot1 * self.deltat
        q2 += qDot2 * self.deltat
        q3 += qDot3 * self.deltat
        q4 += qDot4 * self.deltat
        norm = 1. / sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)    
        self.quat[0] = q1 * norm
        self.quat[1] = q2 * norm
        self.quat[2] = q3 * norm
        self.quat[3] = q4 * norm



        
    def UpdateMadgwickIMU(self):
        """
        madgwickAHRS from https://github.com/xioTechnologies/Open-Source-AHRS-With-x-IMU
        Algorithm IMU update method. Requires only gyroscope and accelerometer data.
        "gx" Gyroscope x axis measurement in radians/s.
        "gy" Gyroscope y axis measurement in radians/s.
        "gz" Gyroscope z axis measurement in radians/s.
        "ax" Accelerometer x axis measurement in any calibrated units.
        "ay" Accelerometer y axis measurement in any calibrated units.
        "az" Accelerometer z axis measurement in any calibrated units.
         
        parameters:
          self.beta
          self.deltat

        input/output:
          self.quaternion
         """
        # short name local variable for readability
        q1 = self.quat[0]
        q2 = self.quat[1]
        q3 = self.quat[2]
        q4 = self.quat[3]   

        ax = self.acc[0]
        ay = self.acc[1]
        az = self.acc[2]
        gx = self.gyr[0]
        gy = self.gyr[1]
        gz = self.gyr[2]


        # Auxiliary variables to avoid repeated arithmetic
        _2q1 = 2f * q1
        _2q2 = 2f * q2
        _2q3 = 2f * q3
        _2q4 = 2f * q4
        _4q1 = 4f * q1
        _4q2 = 4f * q2
        _4q3 = 4f * q3
        _8q2 = 8f * q2
        _8q3 = 8f * q3
        q1q1 = q1 * q1
        q2q2 = q2 * q2
        q3q3 = q3 * q3
        q4q4 = q4 * q4

        # Normalise accelerometer measurement
        norm = sqrt(ax * ax + ay * ay + az * az)
        if (norm == 0f) : raise 
        norm = 1 / norm        
        ax *= norm
        ay *= norm
        az *= norm

        # Gradient decent algorithm corrective step
        s1 = _4q1 * q3q3 + _2q3 * ax + _4q1 * q2q2 - _2q2 * ay
        s2 = _4q2 * q4q4 - _2q4 * ax + 4. * q1q1 * q2 - _2q1 * ay - _4q2 + _8q2 * q2q2 + _8q2 * q3q3 + _4q2 * az
        s3 = 4. * q1q1 * q3 + _2q1 * ax + _4q3 * q4q4 - _2q4 * ay - _4q3 + _8q3 * q2q2 + _8q3 * q3q3 + _4q3 * az
        s4 = 4. * q2q2 * q4 - _2q2 * ax + 4. * q3q3 * q4 - _2q3 * ay
        norm = 1. / sqrt(s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4)    
        s1 *= norm
        s2 *= norm
        s3 *= norm
        s4 *= norm

        # Compute rate of change of quaternion
        qDot1 = 0.5 * (-q2 * gx - q3 * gy - q4 * gz) - self.beta * s1
        qDot2 = 0.5 * (q1 * gx + q3 * gz - q4 * gy) - self.beta * s2
        qDot3 = 0.5 * (q1 * gy - q2 * gz + q4 * gx) - self.beta * s3
        qDot4 = 0.5 * (q1 * gz + q2 * gy - q3 * gx) - self.beta * s4

        # Integrate to yield quaternion
        q1 += qDot1 * self.deltat
        q2 += qDot2 * self.deltat
        q3 += qDot3 * self.deltat
        q4 += qDot4 * self.deltat
        norm = 1. / sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)    
        self.quat[0] = q1 * norm
        self.quat[1] = q2 * norm
        self.quat[2] = q3 * norm
        self.quat[3] = q4 * norm





    def UpdateMahonyAHRS(self, gx, gy, gz, ax,  ay,  az,  mx,  my,  mz):
        """ 
        MahonyAHRS from https://github.com/xioTechnologies/Open-Source-AHRS-With-x-IMU
        Algorithm AHRS update method. Requires gyroscope accelerometer and magnetometer data.
         "gx" Gyroscope x axis measurement in radians/s.
         "gy" Gyroscope y axis measurement in radians/s.
         "gz" Gyroscope z axis measurement in radians/s.
         "ax" Accelerometer x axis measurement in any calibrated units.
         "ay" Accelerometer y axis measurement in any calibrated units.
         "az" Accelerometer z axis measurement in any calibrated units.
         "mx" Magnetometer x axis measurement in any calibrated units.
         "my" Magnetometer y axis measurement in any calibrated units.
         "mz" Magnetometer z axis measurement in any calibrated units.
         
         paremeters:
          self.deltat
          self.kp   Algorithm proportional gain  governs rate of convergence to accelerometer/magnetometer
          self.ki   Algorithm integral gain  governs rate of convergence of gyroscope biases

         input/output: 
           self.quaternion
           self.eInt
        """ 
            
        # short name local variable for readability
        q1 = self.quat[0]
        q2 = self.quat[1]
        q3 = self.quat[2]
        q4 = self.quat[3]

        ax = self.acc[0]
        ay = self.acc[1]
        az = self.acc[2]
        gx = self.gyr[0]
        gy = self.gyr[1]
        gz = self.gyr[2]
        mx = self.mag[0]
        my = self.mag[1]
        mz = self.mag[2]


        # Auxiliary variables to avoid repeated arithmetic
        q1q1 = q1 * q1
        q1q2 = q1 * q2
        q1q3 = q1 * q3
        q1q4 = q1 * q4
        q2q2 = q2 * q2
        q2q3 = q2 * q3
        q2q4 = q2 * q4
        q3q3 = q3 * q3
        q3q4 = q3 * q4
        q4q4 = q4 * q4   

        # Normalise accelerometer measurement
        norm = sqrt(ax * ax + ay * ay + az * az)
        if (norm == 0.) : raise
        norm = 1. / norm        
        ax *= norm
        ay *= norm
        az *= norm

        # Normalise magnetometer measurement
        norm = sqrt(mx * mx + my * my + mz * mz)
        if (norm == 0.)  : raise 
        norm = 1. / norm        
        mx *= norm
        my *= norm
        mz *= norm

        # Reference direction of Earth's magnetic field
        hx = 2. * mx * (0.5. - q3q3 - q4q4) + 2. * my * (q2q3 - q1q4) + 2. * mz * (q2q4 + q1q3)
        hy = 2. * mx * (q2q3 + q1q4) + 2. * my * (0.5 - q2q2 - q4q4) + 2. * mz * (q3q4 - q1q2)
        bx = sqrt((hx * hx) + (hy * hy))
        bz = 2. * mx * (q2q4 - q1q3) + 2. * my * (q3q4 + q1q2) + 2f * mz * (0.5 - q2q2 - q3q3)

        # Estimated direction of gravity and magnetic field
        vx = 2. * (q2q4 - q1q3)
        vy = 2. * (q1q2 + q3q4)
        vz = q1q1 - q2q2 - q3q3 + q4q4
        wx = 2. * bx * (0.5 - q3q3 - q4q4) + 2. * bz * (q2q4 - q1q3)
        wy = 2. * bx * (q2q3 - q1q4) + 2. * bz * (q1q2 + q3q4)
        wz = 2. * bx * (q1q3 + q2q4) + 2. * bz * (0.5 - q2q2 - q3q3)  

        # Error is cross product between estimated direction and measured direction of gravity
        ex = (ay * vz - az * vy) + (my * wz - mz * wy)
        ey = (az * vx - ax * vz) + (mz * wx - mx * wz)
        ez = (ax * vy - ay * vx) + (mx * wy - my * wx)
        if Ki > 0.:
            self.eInt[0] += ex      # accumulate integral error
            self.eInt[1] += ey
            self.eInt[2] += ez
        else:
            self.eInt[0] = 0.0     # prevent integral wind up
            self.eInt[1] = 0.0
            self.eInt[2] = 0.0

        # Apply feedback terms
        gx = gx + self.Kp * ex + self.Ki * self.eInt[0]
        gy = gy + self.Kp * ey + self.Ki * self.eInt[1]
        gz = gz + self.Kp * ez + self.Ki * self.eInt[2]

        # Integrate rate of change of quaternion
        pa = q2
        pb = q3
        pc = q4
        q1 = q1 + (-q2 * gx - q3 * gy - q4 * gz) * (0.5 * self.deltat)
        q2 = pa + (q1 * gx + pb * gz - pc * gy) * (0.5 * self.deltat)
        q3 = pb + (q1 * gy - pa * gz + pc * gx) * (0.5 * self.deltat)
        q4 = pc + (q1 * gz + pa * gy - pb * gx) * (0.5 * self.deltat)

        # Normalise quaternion
        norm = sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)
        norm = 1.0 / norm
        self.quat[0] = q1 * norm
        self.quat[1] = q2 * norm
        self.quat[2] = q3 * norm
        self.quat[3] = q4 * norm


            
    def UpdateMahonyIMU(self):
        """
        MahonyIMU from https://github.com/xioTechnologies/Open-Source-AHRS-With-x-IMU
        Algorithm IMU update method. Requires only gyroscope and accelerometer data.
        "gx" Gyroscope x axis measurement in radians/s.
        "gy" Gyroscope y axis measurement in radians/s.
        "gz" Gyroscope z axis measurement in radians/s.
        "ax" Accelerometer x axis measurement in any calibrated units.
        "ay" Accelerometer y axis measurement in any calibrated units.
        "az" Accelerometer z axis measurement in any calibrated units.
         
        parameters:
          self.beta
          self.deltat

        input/output:
          self.quaternion
         """
        # short name local variable for readability
        q1 = self.quat[0]
        q2 = self.quat[1]
        q3 = self.quat[2]
        q4 = self.quat[3]   
        
        ax = self.acc[0]
        ay = self.acc[1]
        az = self.acc[2]
        gx = self.gyr[0]
        gy = self.gyr[1]
        gz = self.gyr[2]


        #Normalise accelerometer measurement
        norm = sqrt(ax * ax + ay * ay + az * az)
        if (norm == 0.) : raise
        norm = 1. / norm 
        ax *= norm
        ay *= norm
        az *= norm

        # Estimated direction of gravity
        vx = 2.0 * (q2 * q4 - q1 * q3)
        vy = 2.0 * (q1 * q2 + q3 * q4)
        vz = q1 * q1 - q2 * q2 - q3 * q3 + q4 * q4

        # Error is cross product between estimated direction and measured direction of gravity
        ex = (ay * vz - az * vy)
        ey = (az * vx - ax * vz)
        ez = (ax * vy - ay * vx)
        if Ki > 0. :
            self.eInt[0] += ex      # accumulate integral error
            self.eInt[1] += ey
            self.eInt[2] += ez
        else:
            self.eInt[0] = 0.0     # prevent integral wind up
            self.eInt[1] = 0.0
            self.eInt[2] = 0.0

        # Apply feedback terms
        gx = gx + self.Kp * ex + self.Ki * self.eInt[0]
        gy = gy + self.Kp * ey + self.Ki * self.eInt[1]
        gz = gz + self.Kp * ez + self.Ki * self.eInt[2]

        # Integrate rate of change of quaternion
        pa = q2
        pb = q3
        pc = q4
        q1 = q1 + (-q2 * gx - q3 * gy - q4 * gz) * (0.5 * self.deltat)
        q2 = pa + (q1 * gx + pb * gz - pc * gy) * (0.5 * self.deltat)
        q3 = pb + (q1 * gy - pa * gz + pc * gx) * (0.5 * self.deltat)
        q4 = pc + (q1 * gz + pa * gy - pb * gx) * (0.5 * self.deltat)

        # Normalise quaternion
        norm = sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)
        norm = 1.0 / norm
        self.quat[0] = q1 * norm
        self.quat[1] = q2 * norm
        self.quat[2] = q3 * norm
        self.quat[3] = q4 * norm



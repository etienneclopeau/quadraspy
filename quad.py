
import threading
from numpy import array
from pid import PID
import time


matPitch =   array([[ 1, 1],
                    [-1,-1]])
matYaw = array([[ 1,-1],
                [ 1,-1]])
matRoll =   array([[ 1,-1],
                   [-1, 1]])


class Quad():
    def __init__(self, imu, motors, altimeter):

        self.imu = imu
        self.altimeter = altimeter
        self.motors = motors


        self.pid_alt = PID(1,1,1)
        self.pid_roll = PID(1,0,1)
        self.pid_pitch = PID(1,0,1)
        self.pid_yaw = PID(1,0,1)

        self.power_precedent = 0.

        self.setConsigne()

        self.running = True
        threading.Thread(target = self.run).start()
        
    def getPower(self, option = 'test'):
        if option == 'test':
            alt = 1
            power =  0.5

            
        else: 
            alt = self.altimeter.getAltitude()
            power = self.pid_alt.compute(alt, self.alt_c)
        # print 'power',power,alt
        return power

    def getAttitudeRegulation(self, option = 'maintainConsign'):

        roll, pitch, yaw = self.imu.getEuler()

        vmin = -0.5
        vmax = 0.5

        if option == 'maintainConsign':
            regul_roll = self.pid_roll.compute(roll, self.roll_c)
            if regul_roll > vmax : regul_roll = vmax
            if regul_roll < vmin : regul_roll = vmin
            regul_pitch = self.pid_pitch.compute(pitch, self.pitch_c)
            if regul_pitch > vmax : regul_pitch = vmax
            if regul_pitch < vmin : regul_pitch = vmin
            regul_yaw = self.pid_yaw.compute(yaw, self.yaw_c)
            if regul_yaw > vmax : regul_yaw = vmax
            if regul_yaw < vmin : regul_yaw = vmin
            
            dmotors = (regul_roll*matRoll + 1) * (regul_pitch*matPitch + 1) * (regul_yaw*matYaw + 1)
            # print 'attitude',regul_roll,regul_pitch,regul_yaw,dmotors
            return dmotors

        else: raise(notimplementederror)

    def setDistributedPower(self):
        
        power = self.getPower()
        equilibration = self.getAttitudeRegulation()
        
        self.distributedPower = equilibration * power

        # print 'distributedPower', equilibration,power,self.distributedPower

        self.motors.setMotorsSpeed(self.distributedPower)
        

    def setConsigne(self, alt_c = 1, roll_c = 0., yaw_c = 0., pitch_c = 0.):
        self.alt_c = alt_c
        self.roll_c = roll_c
        self.yaw_c = yaw_c
        self.pitch_c = pitch_c
    
    def run(self):
        self.setConsigne()
        while self.running:
            self.setDistributedPower()
            time.sleep(0.1)
    def getDistributedPower(self):
        return self.distributedPower
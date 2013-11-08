# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:32:56 2013

@author: clopeau
"""

import math
import i2c
import time



class hmc5883l:
    # Register map based on Jeff Rowberg <jeff@rowberg.net> source code at
    # https://github.com/jrowberg/i2cdevlib/
    
    HMC5883L_ADDRESS            = 0x1E # this device only has one address
    HMC5883L_DEFAULT_ADDRESS    = 0x1E

    HMC5883L_RA_CONFIG_A        = 0x00
    HMC5883L_RA_CONFIG_B        = 0x01
    HMC5883L_RA_MODE            = 0x02
    HMC5883L_RA_DATAX_H         = 0x03
    HMC5883L_RA_DATAX_L         = 0x04
    HMC5883L_RA_DATAZ_H         = 0x05
    HMC5883L_RA_DATAZ_L         = 0x06
    HMC5883L_RA_DATAY_H         = 0x07
    HMC5883L_RA_DATAY_L         = 0x08
    HMC5883L_RA_STATUS          = 0x09
    HMC5883L_RA_ID_A            = 0x10  #0A
    HMC5883L_RA_ID_B            = 0x11  #0B
    HMC5883L_RA_ID_C            = 0x12  #0C

    HMC5883L_CRA_AVERAGE_BIT    = 6
    HMC5883L_CRA_AVERAGE_LENGTH = 2
    HMC5883L_CRA_RATE_BIT       = 4
    HMC5883L_CRA_RATE_LENGTH    = 3
    HMC5883L_CRA_BIAS_BIT       = 1
    HMC5883L_CRA_BIAS_LENGTH    = 2

    HMC5883L_AVERAGING_1        = 0x00
    HMC5883L_AVERAGING_2        = 0x01
    HMC5883L_AVERAGING_4        = 0x02
    HMC5883L_AVERAGING_8        = 0x03

    HMC5883L_RATE_0P75          = 0x00
    HMC5883L_RATE_1P5           = 0x01
    HMC5883L_RATE_3             = 0x02
    HMC5883L_RATE_7P5           = 0x03
    HMC5883L_RATE_15            = 0x04
    HMC5883L_RATE_30            = 0x05
    HMC5883L_RATE_75            = 0x06

    HMC5883L_BIAS_NORMAL        = 0x00
    HMC5883L_BIAS_POSITIVE      = 0x01
    HMC5883L_BIAS_NEGATIVE      = 0x02

    HMC5883L_CRB_GAIN_BIT       = 7
    HMC5883L_CRB_GAIN_LENGTH    = 3

    HMC5883L_GAIN_1370          = 0x00
    HMC5883L_GAIN_1090          = 0x01
    HMC5883L_GAIN_820           = 0x02
    HMC5883L_GAIN_660           = 0x03
    HMC5883L_GAIN_440           = 0x04
    HMC5883L_GAIN_390           = 0x05
    HMC5883L_GAIN_330           = 0x06
    HMC5883L_GAIN_230           = 0x07

    HMC5883L_MODEREG_BIT        = 1
    HMC5883L_MODEREG_LENGTH     = 2

    HMC5883L_MODE_CONTINUOUS    = 0x00
    HMC5883L_MODE_SINGLE        = 0x01
    HMC5883L_MODE_IDLE          = 0x02

    HMC5883L_STATUS_LOCK_BIT    = 1
    HMC5883L_STATUS_READY_BIT   = 0   
    
    mode = 0

    def __init__(self, port = 1 , address = HMC5883L_DEFAULT_ADDRESS):
        self.bus = i2c.i2c(address)
        self.address = address
        self.initialize()
        
    def initialize(self):
        # write CONFIG_A register
        self.bus.write8(self.HMC5883L_RA_CONFIG_A,
            (self.HMC5883L_AVERAGING_8 << (self.HMC5883L_CRA_AVERAGE_BIT - self.HMC5883L_CRA_AVERAGE_LENGTH + 1)) |
            (self.HMC5883L_RATE_75    << (self.HMC5883L_CRA_RATE_BIT - self.HMC5883L_CRA_RATE_LENGTH + 1)) |
            (self.HMC5883L_BIAS_NORMAL << (self.HMC5883L_CRA_BIAS_BIT - self.HMC5883L_CRA_BIAS_LENGTH + 1)))

        # write CONFIG_B register
        self.setGain(self.HMC5883L_GAIN_1090)
    
        # write MODE register
        self.setMode(self.HMC5883L_MODE_SINGLE)     
        
    def testConnection(self):
        pass
        
    def getSampleAveraging(self):
        pass
    
    def setSampleAveraging(self, value):
        self.bus.writeBits(self.HMC5883L_RA_CONFIG_A, self.HMC5883L_CRA_AVERAGE_BIT, self.HMC5883L_CRA_AVERAGE_LENGTH, value)
        
    def getDataRate(self):
        pass
    
    def setDataRate(self, value):
        self.bus.writeBits(self.HMC5883L_RA_CONFIG_A, self.HMC5883L_CRA_RATE_BIT, self.HMC5883L_CRA_RATE_LENGTH, value)
        
    def getMeasurementBias(self):
        pass
    
    def setMeasurementBias(self, value):
        self.bus.writeBits(self.HMC5883L_RA_CONFIG_A, self.HMC5883L_CRA_BIAS_BIT, self.HMC5883L_CRA_BIAS_LENGTH, value)
        
    def getGain(self):
        pass
        
    def setGain(self, value):
        self.bus.write8(self.HMC5883L_RA_CONFIG_B, value << (self.HMC5883L_CRB_GAIN_BIT - self.HMC5883L_CRB_GAIN_LENGTH + 1))
        
    def getMode(self):
        pass
    
    def setMode(self, newMode):
        # use this method to guarantee that bits 7-2 are set to zero, which is a
        # requirement specified in the datasheet
        self.bus.write8(self.HMC5883L_RA_MODE, newMode << (self.HMC5883L_MODEREG_BIT - self.HMC5883L_MODEREG_LENGTH + 1))
        self.mode = newMode # track to tell if we have to clear bit 7 after a read
     
    #def getAxes(self):
    #    if (self.mode == self.HMC5883L_MODE_SINGLE):
    #        self.bus.write8(self.HMC5883L_RA_MODE, self.HMC5883L_MODE_SINGLE << (self.HMC5883L_MODEREG_BIT - self.HMC5883L_MODEREG_LENGTH + 1))  
    #        time.sleep(0.01)
    #    #packet = self.bus.readBytesListS(self.HMC5883L_RA_DATAX_H, 6)
    #    #packet = self.bus.readList(self.HMC5883L_RA_DATAX_H, 6)
    #       
    #    x,y,z = (packet[0] << 8) | packet[1], (packet[4] << 8) | packet[5],  (packet[2] << 8) | packet[3]
    #        
    #    return x,y,z
 
    def twos_complement(self, val, len):
        # Convert twos compliment to integer
        if (val & (1 << len - 1)):
            val = val - (1<<len)
        return val

    def __convert(self, data, offset):
        val = self.twos_complement(data[offset] << 8 | data[offset+1], 16)
        if val == -4096: return None
        #return round(val * self.__scale, 4)
        return round(val , 4)

    def getRawAxes(self):
        if (self.mode == self.HMC5883L_MODE_SINGLE):
            self.bus.write8(self.HMC5883L_RA_MODE, self.HMC5883L_MODE_SINGLE << (self.HMC5883L_MODEREG_BIT - self.HMC5883L_MODEREG_LENGTH + 1))  
            while not self.getReadyStatus():
                pass
        data = self.bus.bus.read_i2c_block_data(self.address, 0x00)
        #print map(hex, data)
        x = self.__convert(data, 3)
        y = self.__convert(data, 7)
        z = self.__convert(data, 5)
        return x,y,z
       
    def getHeadingX(self):
        # each axis read requires that ALL axis registers be read, even if only
        # one is used this was not done ineffiently in the code by accident
        if (self.mode == self.HMC5883L_MODE_SINGLE):
            self.bus.write8(self.HMC5883L_RA_MODE, self.HMC5883L_MODE_SINGLE << (self.HMC5883L_MODEREG_BIT - self.HMC5883L_MODEREG_LENGTH + 1))  
            time.sleep(0.006)
        packet = self.bus.readBytesListS(self.HMC5883L_RA_DATAX_H, 6)
            
        return ((packet[0] << 8) | packet[1])    
        
    def getHeadingY(self):
        # each axis read requires that ALL axis registers be read, even if only
        # one is used this was not done ineffiently in the code by accident
        if (self.mode == self.HMC5883L_MODE_SINGLE):
            self.bus.write8(self.HMC5883L_RA_MODE, self.HMC5883L_MODE_SINGLE << (self.HMC5883L_MODEREG_BIT - self.HMC5883L_MODEREG_LENGTH + 1))  
            time.sleep(0.006)
        packet = self.bus.readBytesListS(self.HMC5883L_RA_DATAX_H, 6)
           
        return ((packet[4] << 8) | packet[5])    
        
    def getHeadingZ(self):
        # each axis read requires that ALL axis registers be read, even if only
        # one is used this was not done ineffiently in the code by accident
        if (self.mode == self.HMC5883L_MODE_SINGLE):
            self.bus.write8(self.HMC5883L_RA_MODE, self.HMC5883L_MODE_SINGLE << (self.HMC5883L_MODEREG_BIT - self.HMC5883L_MODEREG_LENGTH + 1))  
            time.sleep(0.006)
        packet = self.bus.readBytesListS(self.HMC5883L_RA_DATAX_H, 6)
           
        return ((packet[2] << 8) | packet[3])    
        
    def getLockStatus(self):
        result = self.bus.readBit(self.HMC5883L_RA_STATUS, self.HMC5883L_STATUS_LOCK_BIT)
        return result
        
    def getReadyStatus(self):
        result = self.bus.readBit(self.HMC5883L_RA_STATUS, self.HMC5883L_STATUS_READY_BIT)
        return result
        
    def getIDA(self):
        result = self.bus.readByte(self.HMC5883L_RA_ID_A)
        return result
        
    def getIDB(self):
        result = self.bus.readByte(self.HMC5883L_RA_ID_B)
        return result

    def getIDC(self):
        result = self.bus.readByte(self.HMC5883L_RA_ID_C)
        return result       
    
    def selfTest(self):
    # write CONFIG_A register
        self.bus.write8(self.HMC5883L_RA_CONFIG_A,
            (self.HMC5883L_AVERAGING_8 << (self.HMC5883L_CRA_AVERAGE_BIT - self.HMC5883L_CRA_AVERAGE_LENGTH + 1)) |
            (self.HMC5883L_RATE_75    << (self.HMC5883L_CRA_RATE_BIT - self.HMC5883L_CRA_RATE_LENGTH + 1)) |
            (self.HMC5883L_BIAS_POSITIVE << (self.HMC5883L_CRA_BIAS_BIT - self.HMC5883L_CRA_BIAS_LENGTH + 1)))

        # write CONFIG_B register
        self.setGain(self.HMC5883L_GAIN_660)
    
        # write MODE register
        self.setMode(self.HMC5883L_MODE_SINGLE) 
    
        time.sleep(0.1)
        
        while True:
            magno_x, magno_y, magno_z = self.getAxes()
            print magno_x, magno_y, magno_z 



        
if __name__ == "__main__":      
    logHvalues('maglog.dat',runningtime = 60, sleep = None)      

    #mag = hmc5883l(1, 0x1e)    
    #mag.selfTest()
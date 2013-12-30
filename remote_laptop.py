import Pyro4
import time
#uri=raw_input("What is the Pyro uri of the greeting object? ").strip()

#-- Initialize Pyro client
import Pyro4.naming, Pyro4.core, sys
#Pyro4.core.initClient()

#-- Locate the Name Server
print 'Searching Name Server...'

ns = Pyro4.locateNS(host = '192.168.1.98', port = 9090)
print 'Name Server found '

Pyro4.config.SERIALIZER = 'pickle'
imu=Pyro4.Proxy(ns.lookup("r_IMU"))          # get a Pyro proxy to the greeting object

# while True:
# 	print imu.getEuler()       
# 	time.sleep(0.1)

# import warnings
# warnings.simplefilter('error', UserWarning)


from imu import AnimatedScatter
a = AnimatedScatter(imu)
a.show()
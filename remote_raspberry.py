
#lancer: python -m Pyro4.naming -n 192.168.1.98

import Pyro4
from imu import IMU

imu=IMU()

#Pyro4.naming.startNS(host=None, port=None, enableBroadcast=True, bchost=None, bcport=None, unixsocket=None, nathost=None, natport=None)

daemon=Pyro4.Daemon(host="192.168.1.98")                 # make a Pyro daemon
ns=Pyro4.locateNS()                   # find the name server
uri=daemon.register(imu)   # register the greeting object as a Pyro object
ns.register("r_IMU", uri)  # register the object with a name in the name server

print "Ready. Object uri =", uri      # print the uri so we can use it in the client later
daemon.requestLoop()  
import Pyro4
from imu import IMU

imu=IMU()

daemon=Pyro4.Daemon()                 # make a Pyro daemon
ns=Pyro4.locateNS()                   # find the name server
uri=daemon.register(imu)   # register the greeting object as a Pyro object
ns.register("r_IMU", uri)  # register the object with a name in the name server

print "Ready. Object uri =", uri      # print the uri so we can use it in the client later
daemon.requestLoop()  
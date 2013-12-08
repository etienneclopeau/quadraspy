import Pyro4
from imu import IMU

imu=IMU()

daemon=Pyro4.Daemon()                 # make a Pyro daemon
uri=daemon.register(imu)   # register the greeting object as a Pyro object

print "Ready. Object uri =", uri      # print the uri so we can use it in the client later
daemon.requestLoop()  
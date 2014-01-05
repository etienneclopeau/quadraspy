
#lancer: 
"""export PYRO_SERIALIZER=pickle
export PYRO_SERIALIZERS_ACCEPTED=pickle
python -m Pyro4.naming -n 192.168.1.98 &
sudo python remote_raspberry.py
"""
import Pyro4
from imu import IMU
from capteurs import Altimeter

imu=IMU(algo = 3)
altimeter = Altimeter()
motors = Motors()
quad = Quad(imu,motors,altimeter)


#Pyro4.naming.startNS(host=None, port=None, enableBroadcast=True, bchost=None, bcport=None, unixsocket=None, nathost=None, natport=None)
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
Pyro4.config.SERIALIZER = 'pickle'
daemon=Pyro4.Daemon(host="192.168.1.98")                 # make a Pyro daemon
# daemon=Pyro4.Daemon(host="192.168.1.28")                 # make a Pyro daemon
ns=Pyro4.locateNS()                   # find the name server
uri=daemon.register(imu)   # register the greeting object as a Pyro object
ns.register("r_IMU", uri)  # register the object with a name in the name server

uri=daemon.register(altimeter)   # register the greeting object as a Pyro object
ns.register("r_ALTIMETER", uri)
uri=daemon.register(motors)   # register the greeting object as a Pyro object
ns.register("r_MOTORS", uri)
uri=daemon.register(quad)   # register the greeting object as a Pyro object
ns.register("r_QUAD", uri)


print "Ready. Object uri =", uri      # print the uri so we can use it in the client later
daemon.requestLoop()  

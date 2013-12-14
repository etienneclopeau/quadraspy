import Pyro4

#uri=raw_input("What is the Pyro uri of the greeting object? ").strip()

#-- Initialize Pyro client
import Pyro4.naming, Pyro4.core, sys
#Pyro4.core.initClient()

#-- Locate the Name Server
print 'Searching Name Server...'

ns = Pyro4.locateNS(host = 'raspberrypi', port = 9090)
print 'Name Server found at %s (%s) port %s' % \
 ( ns.URI.address,
 Pyro4.protocol.getHostname(ns.URI.address) or '??',
 ns.URI.port )


imu=Pyro4.Proxy("PYRONAME:r_IMU")          # get a Pyro proxy to the greeting object

print imu.getEuler()          # call method normally

import Pyro4

uri=raw_input("What is the Pyro uri of the greeting object? ").strip()

imu=Pyro4.Proxy(uri)          # get a Pyro proxy to the greeting object
print imu.getEuler()          # call method normally

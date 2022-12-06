import setup_path 
import airsim

import time

# connect to the AirSim simulator
client = airsim.MultirotorClient(ip='192.168.1.104')
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("fly")
client.moveToPositionAsync(0, 0, -10, 5).join()

print("reset")
client.reset()
client.enableApiControl(True)
client.armDisarm(True)
time.sleep(5)
print("done")

print("fly")
client.moveToPositionAsync(0, 0, -10, 5).join()

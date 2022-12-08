import setup_path
import airsim

import sys
import time
from parameters import ip_address

client = airsim.MultirotorClient(ip=ip_address)
client.confirmConnection()
client.reset()
client.enableApiControl(True)
client.armDisarm(True)

print('Taking off')
client.takeoffAsync().join()

# AirSim uses NED coordinates so negative axis is up.
# z of -7 is 7 meters above the original launch point.
# Fly given velocity vector for 5 seconds
z = -7
duration = 5
speed = 1
delay = duration * speed

print('Moving to the indicated altitude')
client.moveToZAsync(z, speed).join()

print("Flying a small square box using moveByVelocityZ")
# using airsim.DrivetrainType.MaxDegreeOfFreedom means we can independently control the drone yaw from the direction
# the drone is flying.  I've set values here that make the drone always point inwards towards the inside of the box (
# which would be handy if you are building a 3d scan of an object in the real world).
vx = speed
vy = 0
print("moving by velocity vx=" + str(vx) + ", vy=" + str(vy) + ", yaw=90")
client.moveByVelocityZAsync(vx, vy, z, duration,
                            airsim.DrivetrainType.MaxDegreeOfFreedom,
                            airsim.YawMode(False, 45)).join()
time.sleep(delay)
vx = 0
vy = speed
print("moving by velocity vx=" + str(vx) + ", vy=" + str(vy) + ", yaw=180")
client.moveByVelocityZAsync(vx, vy, z, duration,
                            airsim.DrivetrainType.MaxDegreeOfFreedom,
                            airsim.YawMode(False, 180)).join()
time.sleep(delay)
vx = -speed
vy = 0
print("moving by velocity vx=" + str(vx) + ", vy=" + str(vy) + ", yaw=270")
client.moveByVelocityZAsync(vx, vy, z, duration,
                            airsim.DrivetrainType.MaxDegreeOfFreedom,
                            airsim.YawMode(False, 270)).join()
time.sleep(delay)
vx = 0
vy = -speed
print("moving by velocity vx=" + str(vx) + ", vy=" + str(vy) + ", yaw=0")
client.moveByVelocityZAsync(vx, vy, z, duration,
                            airsim.DrivetrainType.MaxDegreeOfFreedom,
                            airsim.YawMode(False, 0)).join()
time.sleep(delay)

print('Hovering')
client.hoverAsync().join()

print('Landing')
client.landAsync().join()

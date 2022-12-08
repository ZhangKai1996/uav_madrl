import setup_path
import airsim
from parameters import ip_address

# this script moves the drone to a location, then rests it thousands of time
# purpose of this script is to stress test reset API

# connect to the AirSim simulator 
client = airsim.MultirotorClient(ip=ip_address)
uav_name = 'Drone1'
client.confirmConnection()
client.enableApiControl(True, vehicle_name=uav_name)
client.armDisarm(True, vehicle_name=uav_name)

for idx in range(5):
    state = client.getMultirotorState(vehicle_name=uav_name)
    print(state.kinematics_estimated.linear_velocity)
    client.moveToPositionAsync(0, 0, -10, 5, vehicle_name=uav_name).join()

    state = client.getMultirotorState(vehicle_name=uav_name)
    print(state.kinematics_estimated.linear_velocity)

    client.reset()
    client.enableApiControl(True, vehicle_name=uav_name)
    print("%d" % idx)

# that's enough fun for now. let's quite cleanly
client.enableApiControl(False)

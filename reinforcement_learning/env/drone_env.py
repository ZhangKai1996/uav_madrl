import setup_path
import math
import numpy as np
import time

import airsim
from .airsim_env import AirSimEnv


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape, num_agents=3):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape
        self.n = num_agents
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        self.image_requests = [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
        self.fixed_points = None

    def reset(self):
        fixed_points = [[30.0, 20.0, -50.0],
                        [35.0, 25.0, -50.0],
                        [40.0, 30.0, -50.0]]
        client = self.client
        client.reset()
        n = self.n
        for i in range(n):
            uav_name = 'Drone{}'.format(i + 1)
            client.enableApiControl(True, uav_name)
            client.armDisarm(True, uav_name)
            if i != n - 1:
                client.takeoffAsync(vehicle_name=uav_name)
            else:
                client.takeoffAsync(vehicle_name=uav_name).join()
        self.fixed_points = fixed_points
        print('Reset is completed!')
        return self.__get_obs()

    def step(self, actions):
        client = self.client
        for i, action in enumerate(actions):
            uav_name = 'Drone{}'.format(i + 1)
            state = client.getMultirotorState(vehicle_name=uav_name)
            vel = state.kinematics_estimated.linear_velocity
            client.moveByVelocityAsync(vel.x_val + action[0],
                                       vel.y_val + action[1],
                                       vel.z_val + action[2],
                                       3,
                                       vehicle_name=uav_name)
        time.sleep(3)
        rewards, dones = self.__compute_reward()
        return self.__get_obs(), rewards, dones, {}

    def __get_obs(self):
        obs_n = []
        for i in range(self.n):
            uav_name = 'Drone{}'.format(i + 1)
            response = self.client.simGetImages(self.image_requests, vehicle_name=uav_name)[0]
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            img_rgb = np.reshape(img1d, (response.height, response.width, 3))
            img_rgb = np.flipud(img_rgb)
            image = np.transpose(img_rgb, (2, 0, 1))
            obs_n.append(image)
        return np.stack(obs_n)

    def __compute_reward(self):
        client = self.client
        fixed_points = self.fixed_points

        info, dones = [], []
        for i in range(self.n):
            uav_name = 'Drone{}'.format(i + 1)
            state = client.getMultirotorState(vehicle_name=uav_name)
            collision = client.simGetCollisionInfo(vehicle_name=uav_name).has_collided
            pos = state.kinematics_estimated.position
            info.append([pos, collision])
            dones.append(int(collision))

        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rewards = []
        for i in range(self.n):
            rew = 0
            for f_point in fixed_points:
                dists = []
                for [pos, *_] in info:
                    x_square = math.pow(pos.x_val - f_point[0], 2)
                    y_square = math.pow(pos.y_val - f_point[1], 2)
                    z_square = math.pow(pos.z_val - f_point[2], 2)
                    dist = math.sqrt(x_square + y_square + z_square)
                    dists.append(dist)
                rew -= min(dists) * 0.01

            if info[i][1]:
                rew -= 100
            rewards.append(rew)
        return np.array(rewards), np.array(dones)

    def close(self):
        client = self.client
        for i in range(self.n):
            uav_name = 'Drone{}'.format(i + 1)
            client.armDisarm(False, vehicle_name=uav_name)  # lock
            client.enableApiControl(False, vehicle_name=uav_name)  # release control

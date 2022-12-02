import setup_path
import math
from PIL import Image
import numpy as np

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
        self.image_requests = [airsim.ImageRequest(i, airsim.ImageType.DepthPerspective, True, False)
                               for i in range(5)]
        self.fixed_points = None

    def reset(self):
        self.fixed_points = [[100.0, 100.0, -50.0],
                             [90.0, 100.0, -50.0],
                             [80.0, 100.0, -50.0]]
        client = self.client
        client.reset()
        for i in range(self.n):
            uav_name = 'Drone{}'.format(i + 1)
            client.enableApiControl(True, uav_name)
            client.armDisarm(True, uav_name)

            p = self.fixed_points[i]
            if i != self.n - 1:
                client.moveToPositionAsync(p[0]-60, p[1], p[2], 100, vehicle_name=uav_name)
            else:
                client.moveToPositionAsync(p[0]-60, p[1], p[2], 100, vehicle_name=uav_name).join()
        print('Reset is completed!')
        return self.__get_obs()

    def step(self, actions):
        client = self.client
        for i, action in enumerate(actions):
            uav_name = 'Drone{}'.format(i + 1)
            state = client.getMultirotorState(vehicle_name=uav_name)
            vel = state.kinematics_estimated.linear_velocity
            if i != self.n:
                client.moveByVelocityAsync(vel.x_val + action[0],
                                           vel.y_val + action[1],
                                           vel.z_val + action[2],
                                           5,
                                           vehicle_name=uav_name)
            else:
                client.moveByVelocityAsync(vel.x_val + action[0],
                                           vel.y_val + action[1],
                                           vel.z_val + action[2],
                                           5,
                                           vehicle_name=uav_name).join()
        rewards, dones = self.__compute_reward()
        return self.__get_obs(), rewards, dones, {}

    def __get_obs(self):
        obs_n = []
        for i in range(self.n):
            uav_name = 'Drone{}'.format(i + 1)
            responses = self.client.simGetImages(self.image_requests, vehicle_name=uav_name)
            images = []
            for response in responses:
                img1d = np.array(response.image_data_float, dtype=np.float)
                img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
                img2d = np.reshape(img1d, (response.height, response.width))
                image = Image.fromarray(img2d).convert("L")
                images.append(image)
            obs_n.append(np.stack(images))
        return np.stack(obs_n)

    def __compute_reward(self):
        client = self.client
        fix_points = self.fixed_points
        rewards, dones = [], []
        for i in range(self.n):
            uav_name = 'Drone{}'.format(i + 1)
            state = client.getMultirotorState(vehicle_name=uav_name)
            collision = client.simGetCollisionInfo(vehicle_name=uav_name).has_collided
            pos = state.kinematics_estimated.position
            if not collision:
                point = fix_points[i]
                x_square = math.pow(pos.x_val - point[0], 2)
                y_square = math.pow(pos.y_val - point[1], 2)
                z_square = math.pow(pos.z_val - point[2], 2)
                dist = math.sqrt(x_square + y_square + z_square)
                reward = math.exp(-1 * dist) - 0.5
            else:
                reward = -100
            rewards.append(reward)
            dones.append(int(collision))
        return np.array(rewards), np.array(dones)

    def close(self):
        client = self.client
        for i in range(self.n):
            uav_name = 'Drone{}'.format(i + 1)
            client.armDisarm(False, vehicle_name=uav_name)  # lock
            client.enableApiControl(False, vehicle_name=uav_name)  # release control

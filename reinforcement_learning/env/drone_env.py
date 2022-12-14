import setup_path

import numpy as np
import time
import cv2

import airsim

from .airsim_env import AirSimEnv
from .render import add_ADI
from .util import *


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, image_shape, step_length=1.0, num_agents=3):
        super().__init__(image_shape)
        self.image_shape = image_shape
        self.step_length = step_length
        self.n = num_agents
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        self.image_requests = [airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False), ]
        self.fixed_points = None
        self.limit = 90.0

    def reset(self):
        client = self.client
        client.reset()
        n = self.n
        for i in range(n):
            uav_name = 'Drone{}'.format(i + 1)
            client.enableApiControl(True, uav_name)
            client.armDisarm(True, uav_name)
            if i != n - 1:
                client.moveToZAsync(-10.0, 3, vehicle_name=uav_name)
            else:
                client.moveToZAsync(-10.0, 3, vehicle_name=uav_name).join()

        random_point = np.random.randint(0, 100, size=(n, 3))
        random_point[:, -1] *= -1
        self.fixed_points = random_point
        print('Reset is completed!')
        return self.__get_obs()

    def step(self, actions, duration=3):
        client = self.client
        n = self.n
        step_size = self.step_length
        for i, action in enumerate(actions):
            uav_name = 'Drone{}'.format(i + 1)
            state = client.getMultirotorState(vehicle_name=uav_name).kinematics_estimated
            vel = state.linear_velocity
            # pos = state.position
            # print(uav_name,
            #       '{:>+.3f}, {:>+.3f}, {:>+.3f}'.format(vel.x_val, vel.y_val, vel.z_val),
            #       ["{:>+.3f}".format(a) for a in action], end='\t')
            new_x_val, new_y_val = vel.x_val+action[0]*step_size, vel.y_val+action[1]*step_size
            # print('{:>+.3f}, {:>+.3f}, {:>+.3f}'.format(new_x_val, new_y_val, vel.z_val + action[2]), end='\t')
            # print('{:>+.3f}, {:>+.3f}, {:>+.3f}'.format(pos.x_val, pos.y_val, pos.z_val))
            # print(absolute_bearing((new_x_val, new_y_val)))
            if i != n - 1:
                client.moveByVelocityAsync(new_x_val,
                                           new_y_val,
                                           vel.z_val + action[2],
                                           duration=duration,
                                           # drivetrain=airsim.DrivetrainType.ForwardOnly,
                                           drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                           yaw_mode=airsim.YawMode(False, absolute_bearing((new_x_val, new_y_val))),
                                           vehicle_name=uav_name)
            else:
                client.moveByVelocityAsync(new_x_val,
                                           new_y_val,
                                           vel.z_val + action[2],
                                           duration=duration,
                                           # drivetrain=airsim.DrivetrainType.ForwardOnly,
                                           drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                           yaw_mode=airsim.YawMode(False, absolute_bearing((new_x_val, new_y_val))),
                                           vehicle_name=uav_name).join()
        # print()
        # for i in range(n):
        #     uav_name = 'Drone{}'.format(i + 1)
        #     state = client.getMultirotorState(vehicle_name=uav_name).kinematics_estimated
        #     vel = state.linear_velocity
        #     pos = state.position
        #     print(uav_name,
        #           '{:>+.3f}, {:>+.3f}, {:>+.3f}'.format(vel.x_val, vel.y_val, vel.z_val),
        #           '{:>+.3f}, {:>+.3f}, {:>+.3f}'.format(pos.x_val, pos.y_val, pos.z_val))
        # print()
        # time.sleep(duration)
        rewards, dones = self.__compute_reward()
        return self.__get_obs(), rewards, dones, {}

    def __get_obs(self):
        client = self.client
        image_requests = self.image_requests
        fixed_points = self.fixed_points
        n = self.n
        limit = self.limit

        obs_n, obs_rgb = [], []
        for i in range(n):
            uav_name = 'Drone{}'.format(i + 1)
            state = client.getMultirotorState(vehicle_name=uav_name).kinematics_estimated
            pos = state.position
            vel = state.linear_velocity
            x_y_view_vel = absolute_bearing((vel.x_val, vel.y_val))

            views = []
            for p in fixed_points:
                # y_z_view = bearing((p[1], p[2]), (pos.y_val, pos.z_val))  # x
                x_z_view = relative_bearing((pos.x_val, pos.z_val), (p[0], p[2]))  # y
                x_y_view = relative_bearing((pos.x_val, pos.y_val), (p[0], p[1]), reference=x_y_view_vel)  # z
                views.append([x_y_view, x_z_view])

            response = client.simGetImages(image_requests, vehicle_name=uav_name)[0]
            height, width = response.height, response.width
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            img_rgb = np.reshape(img1d, (height, width, 3))
            img_rgb = add_ADI(i, img_rgb, views, n, width, height, limit)
            obs_rgb.append(img_rgb)
            obs_n.append(np.transpose(img_rgb, (2, 0, 1)))

        # cv2.imshow('Observation', np.hstack(obs_rgb))
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()
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

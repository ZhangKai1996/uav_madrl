import math
import numpy as np
import time
import cv2

import airsim
from .airsim_env import AirSimEnv


def bearing(p1, p0=(0.0, 0.0)):
    delta_x = p1[0] - p0[0]
    delta_y = p1[1] - p0[1]
    if delta_x == 0.0:
        return abs(delta_y) / delta_y * 90.0

    tan_v = delta_y / delta_x
    angle = math.atan(tan_v) * 180.0 / math.pi
    if tan_v > 0:
        angle += int(delta_y < 0) * 180.0
    else:
        angle -= int(delta_y > 0) * 180.0
    angle = (angle + 360) % 360
    if angle >= 180.0:
        angle -= 360
    return angle


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape, num_agents=3):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape
        self.n = num_agents
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        self.image_requests = [airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
                               airsim.ImageRequest("front_left", airsim.ImageType.Scene, False, False),
                               airsim.ImageRequest("bottom_center", airsim.ImageType.Scene, False, False)]
        self.fixed_points = None

        self.limit = 60.0

    def reset(self):
        client = self.client
        client.reset()
        n = self.n
        fixed_points = []
        for i in range(n):
            uav_name = 'Drone{}'.format(i + 1)
            client.enableApiControl(True, uav_name)
            client.armDisarm(True, uav_name)
            if i != n - 1:
                client.moveToZAsync(-10.0, 3, vehicle_name=uav_name)
            else:
                client.moveToZAsync(-10.0, 3, vehicle_name=uav_name).join()
            random_point = np.random.randint(0, 100, size=(3,))
            random_point[-1] = -50.0
            fixed_points.append(list(random_point))
        self.fixed_points = fixed_points
        print('Reset is completed!')
        return self.__get_obs()

    def step(self, actions):
        client = self.client
        for i, action in enumerate(actions):
            uav_name = 'Drone{}'.format(i + 1)
            vel = client.getMultirotorState(vehicle_name=uav_name).kinematics_estimated.linear_velocity
            new_x_val, new_y_val = vel.x_val + action[0] * 2, vel.y_val + action[1] * 2
            client.moveByVelocityAsync(new_x_val,
                                       new_y_val,
                                       vel.z_val + action[2] * 2,
                                       duration=5,
                                       drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                       yaw_mode=airsim.YawMode(False, bearing((new_x_val, new_y_val))),
                                       vehicle_name=uav_name)
        time.sleep(5)
        rewards, dones = self.__compute_reward()
        return self.__get_obs(), rewards, dones, {}

    def __get_obs(self):
        client = self.client
        image_requests = self.image_requests
        fixed_points = self.fixed_points
        n = self.n
        limit = self.limit

        obs_n = []
        for i in range(n):
            uav_name = 'Drone{}'.format(i + 1)
            pos = client.getMultirotorState(vehicle_name=uav_name).kinematics_estimated.position
            views = []
            for p in fixed_points:
                # y_z_view = bearing((p[1], p[2]), (pos.y_val, pos.z_val))  # x
                x_z_view = bearing((p[0], p[2]), (pos.x_val, pos.z_val))  # y
                x_y_view = bearing((p[0], p[1]), (pos.x_val, pos.y_val))  # z
                # views.append([x_y_view, x_z_view, y_z_view])
                views.append([x_y_view, x_z_view])

            response = client.simGetImages(image_requests, vehicle_name=uav_name)[0]
            height, width = response.height, response.width
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            img_rgb = np.reshape(img1d, (height, width, 3))
            # img_rgb = cv2.ellipse(img_rgb,
            #                       (width // 2, height // 2),
            #                       (width // 2 - 10, height // 2 - 10),
            #                       0, 0, 360,
            #                       (0, 255, 0),
            #                       1)
            center_x, center_y = width // 2, height // 2
            width_r, height_r = width // 2, height // 6
            radius = int(height_r / (n-1) / 2)
            # horizontal rectangle
            img_rgb = cv2.rectangle(img_rgb,
                                    (center_x - width_r // 2, height - 10 - height_r),
                                    (center_x + width_r // 2, height - 10),
                                    (0, 255, 0), 1)
            # vertical rectangle
            img_rgb = cv2.rectangle(img_rgb,
                                    (width - 10 - height_r, center_y - width_r // 2),
                                    (width - 10, center_y + width_r // 2),
                                    (0, 255, 0), 1)
            img_rgb = cv2.line(img_rgb,
                               (center_x, height - 10 - height_r),
                               (center_x, height - 10),
                               (0, 255, 0), 1)
            img_rgb = cv2.line(img_rgb,
                               (width - 10 - height_r, center_y),
                               (width - 10, center_y),
                               (0, 255, 0), 1)
            for [horizontal, vertical] in views:
                if abs(horizontal) >= limit:
                    horizontal = abs(horizontal) / horizontal * limit
                delta = int(horizontal / limit * width_r / 2)
                img_rgb = cv2.circle(img_rgb,
                                     (center_x+delta, height-10-height_r//2),
                                     radius,
                                     (0, 0, 255), 1)

                if abs(vertical) >= limit:
                    vertical = abs(vertical) / vertical * limit
                delta = int(vertical / limit * height_r / 2)
                img_rgb = cv2.circle(img_rgb,
                                     (width-10-height_r//2, center_y+delta),
                                     radius,
                                     (0, 0, 255), 1)
            image = np.transpose(img_rgb, (2, 0, 1))
            if uav_name == 'Drone1':
                cv2.imshow(uav_name, img_rgb)
                cv2.waitKey(1)
                cv2.destroyAllWindows()
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

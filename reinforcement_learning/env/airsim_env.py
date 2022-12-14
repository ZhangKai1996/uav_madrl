import setup_path

import gym
from gym import spaces
import numpy as np


class AirSimEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, image_shape):
        (width, height, channel) = image_shape
        self.observation_space = spaces.Box(0.0, 1.0, shape=(width*height*channel,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
        self.viewer = None

    def close(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def render(self):
        raise NotImplementedError()

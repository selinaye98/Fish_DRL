import math
import re
import time
import torch
import io
import numpy as np
import test_2d_flow_stream_around_fish_pybind as fish
import gymnasium as gym
from gymnasium import spaces

class FISHTestEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self.alpha = 0.1
        self.episode = 100001
        self.time_per_action = 0.1
        low_action = np.array([-1, -1]).astype(np.float32)
        high_action = np.array([1, 1]).astype(np.float32)
        low_obs = np.full(18, 0).astype(np.float32)
        high_obs = np.full(18, 10).astype(np.float32)
        self.arr = np.full(18, 100)

        self.action_space = spaces.Box(low_action, high_action)
        self.observation_space = spaces.Box(low_obs, high_obs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.total_reward = 0.0
        self.action_time = 0.0
        self.action_time_steps = 0
        self.frequency = 3
        self.lamb = 2

        self.fish = fish.from_sph_cpp(self.episode)
        self.fish.SetFreq(self.frequency)
        self.fish.SetLambda(self.lamb)
        self.fish.RunCase(self.episode, self.action_time)
        self.fish_start_x = self.fish.GetFishHeadPositionX()
        self.fish_start_y = self.fish.GetFishHeadPositionY()
        print('x', self.fish_start_x)
        print('y', self.fish_start_y)

        obs = np.array(
            [self.fish.GetPressurePoint1(), self.fish.GetPressurePoint2(), self.fish.GetPressurePoint3(),
             self.fish.GetPressurePoint4(), self.fish.GetPressurePoint5(), self.fish.GetPressurePoint6(),
             self.fish.GetPressurePoint7(), self.fish.GetPressurePoint8(), self.fish.GetPressurePoint9(),
             self.fish.GetPressurePoint11(), self.fish.GetPressurePoint12(), self.fish.GetPressurePoint13(),
             self.fish.GetPressurePoint14(), self.fish.GetPressurePoint15(), self.fish.GetPressurePoint16(),
             self.fish.GetPressurePoint17(), self.fish.GetPressurePoint18(), self.fish.GetPressurePoint19()]) / self.arr
        self._get_obs = obs.astype(np.float32)
        print('test_obs: ', self._get_obs)

        return self._get_obs, {}

    def step(self, action):  
        self.action_time_steps += 1
        self.frequency_new = action[0] + 3.0
        self.lamb_new = action[1] + 2.0
        print('test_fre: ', self.frequency_new)
        print('test_lam: ', self.lamb_new)

        file = open(f'action_{self.episode}.txt', 'a')
        file.write('action_time:  ')
        file.write(str(self.action_time))
        file.write('  ferquency:  ')
        file.write(str(self.frequency_new))
        file.write('  lamb:  ')
        file.write(str(self.lamb_new))
        file.write('\n')
        
        self.fish_previous_x = self.fish.GetFishHeadPositionX() - self.fish_start_x
        self.fish_previous_y = self.fish.GetFishHeadPositionY() - self.fish_start_y
        
        for i in range(25):
            self.frequency = self.frequency + self.alpha * (self.frequency_new - self.frequency)
            self.lamb = self.lamb + self.alpha * (self.lamb_new - self.lamb)
            self.fish.SetFreq(self.frequency)
            self.fish.SetLambda(self.lamb)
            self.action_time += self.time_per_action / 25
            self.fish.RunCase(self.episode, self.action_time)

        self.fish_now_x = self.fish.GetFishHeadPositionX() - self.fish_start_x
        self.fish_now_y = self.fish.GetFishHeadPositionY() - self.fish_start_y

        punish = 10 * (self.fish_now_y) ** 2
        award = self.fish.GetFishHeadVelocityX() # + abs(self.fish_now_x - self.fish_previous_x)

        print('punish: ', punish)
        print('award: ', award)
        reward = award - punish

        obs = np.array(
            [self.fish.GetPressurePoint1(), self.fish.GetPressurePoint2(), self.fish.GetPressurePoint3(),
             self.fish.GetPressurePoint4(), self.fish.GetPressurePoint5(), self.fish.GetPressurePoint6(),
             self.fish.GetPressurePoint7(), self.fish.GetPressurePoint8(), self.fish.GetPressurePoint9(),
             self.fish.GetPressurePoint11(), self.fish.GetPressurePoint12(), self.fish.GetPressurePoint13(),
             self.fish.GetPressurePoint14(), self.fish.GetPressurePoint15(), self.fish.GetPressurePoint16(),
             self.fish.GetPressurePoint17(), self.fish.GetPressurePoint18(), self.fish.GetPressurePoint19()]) / self.arr
        self._get_obs = obs.astype(np.float32)

        award_out = 0.0
        punish_out = 0.0
        if self.action_time_steps > 199 or abs(self.fish_now_x) >= 1.92:
            done = True
            award_out = 1.0
            reward = reward + award_out
        elif abs(self.fish_now_y) >= 0.1:
            done = True
            punish_out = - 100
            reward = reward + punish_out
        elif self.action_time_steps > 10 or abs(self.fish_now_x) >= 1.92:
            done = True
            award_out = 1.0
            reward = reward + award_out
        else:
            done = False

        file = open(f'reward_{self.episode}.txt', 'a')
        file.write('action_time:  ')
        file.write(str(self.action_time))
        file.write('  reward:  ')
        file.write(str(reward))
        file.write('  award  ')
        file.write(str(award))
        file.write('  punish:  ')
        file.write(str(punish))
        file.write('  award_out:  ')
        file.write(str(award_out))
        file.write('  punish_out:  ')
        file.write(str(punish_out))
        file.write('\n')
        file.close()

        self.total_reward += reward

        if done == True:
            file = open('reward_test.txt', 'a')
            file.write('episode:  ')
            file.write(str(self.episode))
            file.write('  reward:  ')
            file.write(str(self.total_reward))
            file.write('\n')
            file.close()
            self.episode += 1

        return self._get_obs , reward, done, False, {}

    def render(self):
        return 0

    def _render_frame(self):
        return 0

    def close(self):
        return 0

# from src.agents.agent import Agent, DecisionMaker
# from src.environments.env_wrapper import*
# 'Environment Related Imports'
# import tqdm
# import gym
import PIL
import matplotlib.pyplot as plt
import time

from gym import Wrapper
from gym.spaces import MultiDiscrete, Box


# 'Deep Model Related Imports'
from torch.nn.functional import one_hot
import torch
import numpy as np
from stable_baselines3 import DQN, PPO

from environments.env_wrapper import BattleFieldSingleEnv,CreateEnvironment
from utils.functions import CreateDecentralizedAgents, CreateCentralizedController, \
    CreateDecentralizedController

# del model # remove to demonstrate saving and loading
#
# model = DQN.load("dqn_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()
from DMs.simple_DMs import Do_action_DM, DecisionMaker, Stay_DM


class PPODecisionMaker(DecisionMaker):
    def __init__(self, action_space):
        self.space = action_space
        self.model_file_name = 'BF_PPO1'
        try:
            self.model = PPO.load(self.model_file_name)
        except:
            self.train_model()
        # print(f"{self.model.observation_space}")

        self.__is_image_obs = isinstance(self.model.observation_space, Box)

        if self.__is_image_obs:
            self.obs_size = np.prod(self.model.observation_space.shape)
        elif isinstance(self.model.observation_space, MultiDiscrete):
            self.obs_size = len(self.model.observation_space.nvec)
        else:
            raise NotImplemented('other observation spaces not supported')


        # print(f"{self.obs_size}")
        # for obs in self.model.observation_space:
        #     print(f"obs size: {obs}")
        #     print(f"obs size: { obs.n }")

# fix observation space if not fit to trained dimension
    def fit_obs(self, obs):
        temp = [obj for obj in obs[:(self.obs_size-1)]]
        temp.append(obs[-1])
        print(f"{temp}")
        return temp

    def get_action(self, observation):
        if not self.__is_image_obs:
            if (len(observation)!=self.obs_size):
                observation = self.fit_obs(observation)
        action, _states = self.model.predict(observation, deterministic=True)
        return action

    def retrain(self,env):
        self.model.set_env(env)
        self.model.learn(total_timesteps=200000, n_eval_episodes=200)
        self.model.save(self.model_file_name)



    def train_model(self):
        env = CreateEnvironment()
        agent = "blue_0"
        temp_env = BattleFieldSingleEnv(env, Stay_DM, Stay_DM, agent)

        obs = temp_env.reset()
        # for i in range(20):
        #     single_env.step(15)
        #     single_env.render()
        #     time.sleep(0.2)

        temp_env.render()

        model = PPO("MlpPolicy", temp_env, verbose=1)
        model.learn(total_timesteps=2000000,n_eval_episodes=200)
        self.model = model
        self.model.save(self.model_file_name)

# def from_RGBarray_to_image(obs):
#     fig = plt.figure(figsize=(16, 4))
#     for ob in obs:
#         plt.imshow(obs)
#         # if filename is None:
#         plt.show()
#     # ax = fig.add_subplot(1, len(self.agents), i)
#     # i += 1
#     # plt.title(title)
#     # ax.imshow(observation[agent_name])
#     return PIL.Image.frombytes('RGB',
#                         fig.canvas.get_width_height(), fig.canvas.tostring_rgb())



if __name__ == '__main__':
    # check code:
    env = CreateEnvironment()
    agent = "blue_0"
    single_env = BattleFieldSingleEnv(env, Stay_DM(env.action_spaces[agent],6), Stay_DM(env.action_spaces[agent],0), agent)



    # obs = (obs[a_n] for a_n in agents)
    # im = from_RGBarray_to_image(obs)
    D_M = PPODecisionMaker(single_env.action_space)
    #single_env.reset()
    #D_M.retrain(single_env)

    obs = single_env.reset()
    total_reward = 0
    for i in range(100):
        next_a = D_M.get_action(obs)
        obs, rew, done, _ = single_env.step(next_a)
        total_reward+=rew
        print(f"action: {next_a}, reward: {rew}, total rew: {total_reward}")
        single_env.render()
        time.sleep(0.1)


    # print(f"obs:{obs}")
    # print(f"next action: { env.index_action_dictionary[D_M.get_action(obs)]}")




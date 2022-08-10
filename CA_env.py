import time

from pettingzoo.magent import battlefield_v5
from random import sample
# Just to silent an harmless warning
from warnings import filterwarnings

from agents.agent import Do_action_DM, Stay_DM
from DMs.simple_planner import Simple_DM
from copy import deepcopy

env = battlefield_v5.env(map_size=46)

obs = env.reset()

env.render()

print('env created')


from agents import Agent, DecisionMaker, RandomDecisionMaker
from control import CentralizedController, DecentralizedController
from environments import EnvWrapperPZ
import copy

# import pandas as pd
# import seaborn as sns

filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')


# A Wrapper for the pettingzoo environment within MAC
class BattleFieldEnv(EnvWrapperPZ):
    def __init__(self, env):
        super().__init__(env)
        self.obs = self.env.reset()
        # print(self.obs)

    def step(self, joint_action):
        return self.env.step(joint_action)

    def observation_to_dict(self, obs):
        return obs

    def reset(self):
        return self.env.reset()

class BattleFieldSingleEnv():
    def __init__(self, env,blue_other_dm, red_DM, agent="blue_0"):
        self.env = env
        self.action_space = env.action_spaces[agent]
        self.observation_space = env.observation_spaces[agent]
        self.agent_name = agent
        self.agents = CreateDecentralizedAgents(env, blue_other_dm, red_DM)
        self.others_blue_DM = blue_other_dm
        self.others_red_DM = red_DM
        self.all_obs = self.env.reset()
        self.obs = self.all_obs[agent]
        self.metadata = None
        # print(self.obs)


    def step(self, action):
        joint_action = {}
        for agent_name in self.all_obs.keys():
            act = self.agents[agent_name].get_decision_maker().get_action(self.all_obs[agent_name])
            joint_action[agent_name] = act
        joint_action[self.agent_name] = action
        self.all_obs, reward, done, self.metadata = self.env.step(joint_action)
        try:
            self.obs = self.all_obs[self.agent_name]
        except:
            return (None, None, True, self.metadata)
        return (self.obs, reward[self.agent_name], done[self.agent_name], self.metadata)

    def observation_to_dict(self, obs):
        return obs

    def render(self):
        return self.env.render()

    def reset(self):
        self.all_obs = self.env.reset()
        self.obs = self.all_obs[self.agent_name]
        return self.obs

    def print_env_info(self):
        for i, agent in enumerate(self.env.agents, 1):
            print(f'- agent {i}: {agent}')
            print(f'\t- observation space: {self.env.observation_space(agent)}')
            print(f'\t- action space: {self.env.action_space(agent)}')
            print(f'\t- action space sample: {self.env.action_space(agent).sample()}')


class Coordinator:
    def __init__(self):
        pass


class Dashboard:
    def __init__(self, env):
        pass



def CreateEnvironment(minimap=False):
    # Create and reset PettingZoo environment
    BF_env = battlefield_v5.parallel_env(map_size=46, minimap_mode=minimap, step_reward=-0.005, dead_penalty=-0.1,
                                         attack_penalty=-0.01, attack_opponent_reward=0.5, max_cycles=1000,
                                         extra_features=False)
    BF_env.reset()

    # Create a MAC from the PZ environment
    return BattleFieldEnv(BF_env)


class GreedyDecisionMaker(DecisionMaker):
    def __init__(self, env):
        # We create a full copy of the environment to use it for planning but without changing the original environment
        self.env = copy.deepcopy(env)

    def get_action(self, observation):

        if isinstance(self.space, dict):
            return {agent: self.space[agent].sample() for agent in self.space.keys()}
        else:
            return self.space.sample()


# Create a random agent using a random decision maker
def CreateRandomAgent(env):
    return Agent(RandomDecisionMaker(env.action_spaces))

# Create an agent that repeats an action all the time
def createOneActionAgent(action):
    return Agent(Do_action_DM(action))



# Create identical agents with the same decision maker
def CreateDecentralizedIdenticalAgents(env, decision_maker):
    decentralized_agents = {
        agent_id: Agent(decision_maker(env.action_spaces[agent_id]))
        for agent_id in mac_BF_env.get_env_agents()
    }
    return decentralized_agents


# Create multiple agents divided into two groups of different decision makers
def CreateDecentralizedAgents(env, blue_decision_maker, red_decision_maker):
    decentralized_blue_agents = {
        agent_id: Agent(deepcopy(blue_decision_maker))
        for agent_id in env.get_env_agents() if 'blue' in agent_id
    }
    # decentralized_blue_agents["blue_0"] = Agent(blue_decision_maker(env.action_spaces["blue_0"],10))
    # decentralized_blue_agents["blue_1"] = Agent(blue_decision_maker(env.action_spaces["blue_1"], 14))

    # decentralized_red_agents = {
    #     agent_id: Agent(red_decision_maker(env.action_spaces[agent_id]))
    #     for agent_id in mac_BF_env.get_env_agents() if 'red' in agent_id
    # }

    decentralized_red_agents = {
        agent_id: Agent(deepcopy(red_decision_maker))
        for agent_id in env.get_env_agents() if 'red' in agent_id
    }

    merged_dict = {**decentralized_blue_agents, **decentralized_red_agents}
    return merged_dict


# Create and run a centralized controller using a given agent
def CreateCentralizedController(env, agent):
    # Creating a centralized controller with the random agent
    centralized_random_controller = CentralizedController(env, agent)

    # Running the centralized random agent
    centralized_random_controller.run(render=True, max_iteration=1000)


#Create and run a decentralized controller using a given dictionary of agents
def CreateDecentralizedController(env, agents):
    # Creating a decentralized controller with the random agents
    decentralized_random_controller = DecentralizedController(env, agents)

    # Running the decentralized agents
    decentralized_random_controller.run(render=True, max_iteration=1000)


## Main
if __name__ == '__main__':
    env = CreateEnvironment()
    agent = "blue_11"
    action_space = env.action_spaces[agent]
    temp_env = BattleFieldSingleEnv(env, Simple_DM(action_space,0.5), Simple_DM(action_space,0.5,red_team=True), agent)

    obs = temp_env.reset()

    simple_dm = Simple_DM(temp_env.action_space)

    total_reward = 0
    for i in range(100):
        a = simple_dm.get_action(obs)
        obs,rew,done,_ = temp_env.step(a)
        if done:
            break
        temp_env.render()
        total_reward+=rew
        print(f"action: {a}, reward: {rew}, total rew: {total_reward}")
        time.sleep(0.3)

    print(f" total rew: {total_reward}")





    # mac_BF_env = CreateEnvironment()

    # CreateCentralizedController(mac_BF_env, CreateRandomAgent(mac_BF_env))

    # CreateDecentralizedController(mac_BF_env, CreateDecentralizedIdenticalAgents(mac_BF_env, RandomDecisionMaker))

    # CreateDecentralizedController(mac_BF_env, CreateDecentralizedAgents(mac_BF_env, Stay_DM , Stay_DM))

    # CreateDecentralizedController(mac_BF_env, CreateDecentralizedAgents(mac_BF_env, RandomDecisionMaker, RandomDecisionMaker))


    # GDM = GreedyDecisionMaker(mac_BF_env)

    # GDM.get_action()






# # dont use this
# a = env.action_space.sample()
# ret = env.step(a)
# print(f"{ret}")
#
# # # Make sure it works with our API:
# env.agents = env.taxis_names
# # print(f"{env.agents}\n")
# env.action_spaces = {
#     agent_name: env.action_space for agent_name in env.agents
# }
# env.observation_spaces = {
#     agent_name: env.observation_space for agent_name in env.agents
# }
# env.possible_agents = [agent for agent in env.agents]
# #
#
#
# #
# print('EnvironmentWrapper created')
#
# a = env.action_space.sample()
# ret = env.step(a)
# print(f"{ret}")
#
# # path = environment.plan_drop_only([0,0],[2,3],4)
# # print(f"{path}")
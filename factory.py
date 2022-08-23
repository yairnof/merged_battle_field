from pettingzoo.magent import battlefield_v5
from pettingzoo.magent import battle_v4
from agents import Agent
from DMs.simple_DMs import *
from control import CentralizedController, DecentralizedController
from control.cont_decentral_coordinator import DecentralizedControllerCoordinator
from environments import EnvWrapperPZ
import constants as const
from copy import deepcopy


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

    def print_env_info(self):
        for i, agent in enumerate(self.env.agents, 1):
            print(f'- agent {i}: {agent}')
            print(f'\t- observation space: {self.env.observation_space(agent)}')
            print(f'\t- action space: {self.env.action_space(agent)}')
            print(f'\t- action space sample: {self.env.action_space(agent).sample()}')


def CreateEnvironment():
    #Create and reset PettingZoo environment
    # BF_env = battlefield_v5.parallel_env(map_size=const.MAP_SIZE, minimap_mode=True, step_reward=-0.005, dead_penalty=-0.1,
    #                                      attack_penalty=-0.1, attack_opponent_reward=0.2, max_cycles=1000,
    #                                      extra_features=True)

    BF_env = battle_v4.parallel_env(map_size=const.MAP_SIZE, minimap_mode=True, step_reward=-0.005, dead_penalty=-0.1,
                                         attack_penalty=-0.1, attack_opponent_reward=0.2, max_cycles=1000,
                                         extra_features=True)
    BF_env.reset()

    # Create a MAC from the PZ environment
    return BattleFieldEnv(BF_env)


# Create a random agent using a random decision maker
def CreateRandomAgent(env):
    return Agent(RandomDecisionMaker(env.action_spaces))


# Create identical agents with the same decision maker
def CreateDecentralizedIdenticalAgents(env, decision_maker):
    decentralized_agents = {
        agent_id: Agent(decision_maker(env.action_spaces[agent_id]))
        for agent_id in env.get_env_agents()
    }
    return decentralized_agents


# Create multiple agents divided into two groups of different decision makers
def CreateDecentralizedAgents_old(env, blue_decision_maker, red_decision_maker):
    decentralized_blue_agents = {
        agent_id: Agent(blue_decision_maker(env.action_spaces[agent_id]))
        for agent_id in env.get_env_agents() if 'blue' in agent_id
    }

    decentralized_red_agents = {
        agent_id: Agent(red_decision_maker(env.action_spaces[agent_id]))
        for agent_id in env.get_env_agents() if 'red' in agent_id
    }

    merged_dict = {**decentralized_blue_agents, **decentralized_red_agents}
    return merged_dict


# Create multiple agents divided into two groups of different decision makers
def CreateDecentralizedAgents(env, blue_decision_maker, red_decision_maker, blue_use_env_and_ids=False, red_use_env_and_agent_ids=False):
    if red_use_env_and_agent_ids:
        decentralized_red_agents = {
            agent_id: Agent(red_decision_maker(env, agent_id))
            for agent_id in env.get_env_agents() if 'red' in agent_id
        }
    else:
        decentralized_red_agents = {
            agent_id: Agent(deepcopy(red_decision_maker))
            for agent_id in env.get_env_agents() if 'red' in agent_id
        }

    if blue_use_env_and_ids:
        decentralized_blue_agents = {
            agent_id: Agent(blue_decision_maker(env, agent_id))
            for agent_id in env.get_env_agents() if 'blue' in agent_id
        }
    else:
        decentralized_blue_agents = {
            agent_id: Agent(deepcopy(blue_decision_maker))
            for agent_id in env.get_env_agents() if 'blue' in agent_id
        }

    merged_dict = {**decentralized_blue_agents, **decentralized_red_agents}
    return merged_dict


# Create and run a centralized controller using a given agent
def CreateCentralizedController(env, agent):
    # Creating a centralized controller with the random agent
    centralized_controller = CentralizedController(env, agent)

    # Running the centralized random agent
    centralized_controller.run(render=True, max_iteration=1000)


# Create and run a decentralized controller using a given dictionary of agents
def CreateDecentralizedController(env, agents, coordinator=None, plan_length=0):
    # Creating a decentralized controller with the random agents
    decentralized_controller = DecentralizedControllerCoordinator(env, agents, coordinator, plan_length)

    # Running the decentralized agents
    decentralized_controller.run(render=True, max_iteration=const.MAX_ITERATIONS)


# Create a simulation of joint_plan
def CreateSimulationController(env, joint_plan):
    sim_controller = CentralizedController(env, Agent(SimDecisionMaker(joint_plan=joint_plan)))
    plan_length = min([len(plan) for (agent, plan) in joint_plan.items()])
    sim_controller.run(render=False, max_iteration=plan_length)
    return sim_controller.total_rewards, sim_controller.observations




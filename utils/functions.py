# assisting functions
from copy import deepcopy
from agents import Agent
from DMs.simple_DMs import Do_action_DM, RandomDecisionMaker
from control import CentralizedController, DecentralizedController



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
        for agent_id in env.get_env_agents()
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



from DMs.simple_DMs import *
import factory
import coordinator
import performance
import constants as const
from DMs.simple_planner import Simple_DM

import numpy as np
from numpy.random import default_rng
import networkx as nx
import matplotlib.pyplot as plt
from DMs.simple_DMs import AttackNearestEnemy
from DMs.simple_planner import ApproxBestAction

# Test a centralized controller with a random decision maker
def test_centralized_controller(env):
    factory.CreateCentralizedController(env, factory.CreateRandomAgent(env))


# Test a decentralized controller with a random decision maker
def test_decentralized_controller(env):
    factory.CreateDecentralizedController(env, factory.CreateDecentralizedAgents(env, RandomDecisionMaker(env.action_spaces['red_1']), RandomDecisionMaker(env.action_spaces['blue_1'])))


# Test a simulator controller (Only runs a predefined joint plan) with random decision makers
def test_sim_controller(env):
    RDM_dict = {agent_id: RandomDecisionMaker(env.action_spaces[agent_id]) for agent_id in env.get_env_agents()}
    plan_dict = {agent_id: rdm.get_plan(None, const.LONG_PLAN_LENGTH) for (agent_id, rdm) in RDM_dict.items()}
    total_rewards, observations = factory.CreateSimulationController(env, plan_dict)
    result = performance.objective(plan_dict, observations, total_rewards)
    print(result)


# Test a coordinator within a decentralized controller, using random decision makers.
# The IdentityCoordinator approves every plan as is
def test_coordinator(env):
    factory.CreateDecentralizedController(env, factory.CreateDecentralizedAgents(env, RandomDecisionMaker(env.action_spaces['red_1']), RandomDecisionMaker(env.action_spaces['blue_1'])), coordinator=coordinator.IdentityCoordinator(env), plan_length=const.PLAN_LENGTH)


# Test a coordinator within a decentralized controller.
# The SimGreedyCoodinator uses a simulator controller to simulate the joint plan,
# and it uses a greedy mechanism and a binary hard constraint between every pair of plans to decide which plan to take.
# The unapproved plans become plans with a repeated default action (Do nothing)
def test_sim_coordinator(env):
    factory.CreateDecentralizedController(env, factory.CreateDecentralizedAgents(env, RandomDecisionMaker(env.action_spaces['red_1']), RandomDecisionMaker(env.action_spaces['blue_1'])), coordinator=coordinator.SimGreedyCoordinator(env), plan_length=const.PLAN_LENGTH)


# Test simple decision maker without a coordinator
def test_simple_dm(env):
    action_space = env.action_spaces['blue_11']
    factory.CreateDecentralizedController(env, factory.CreateDecentralizedAgents(env, Simple_DM(action_space, 0.5), Simple_DM(action_space, 0.5, red_team=True)))


# Test simple decision maker with a greedy coordinator
def test_simple_dm_coordinated(env):
    action_space = env.action_spaces['blue_11']
    factory.CreateDecentralizedController(env, factory.CreateDecentralizedAgents(env, Simple_DM(action_space, 0.5), Simple_DM(action_space, 0.5, red_team=True)), coordinator=coordinator.SimGreedyCoordinator(env), plan_length=1)


# Test the building of a grid graph and several queries
def test_grid_graph():
    SIZE = 13
    agent_map = default_rng(42).random((SIZE, SIZE))
    agent_map = np.round_(agent_map)
    plt.imshow(agent_map)
    plt.show()
    gg = dm.build_grid_graph(agent_map, SIZE)

    try:
        path = nx.dijkstra_path(gg, "0:10", "8:8", weight="weight")
        path_length = nx.path_weight(gg, path, weight='weight')
        multi_source_length, multi_source_path = nx.multi_source_dijkstra(gg, {"10:0", "12:9"}, "8:8")
        print(path, path_length, multi_source_path, multi_source_length)
        node_colors = ["blue" if n in path else "red" for n in gg.nodes()]
        nx.draw(gg, nx.get_node_attributes(gg, 'pos'), node_color=node_colors, with_labels=True, node_size=10)
        plt.show()
        return path, path_length, multi_source_path, multi_source_length
    except Exception as e:
        print(e)


def test_attack_nearest(env):
    obj = factory.CreateDecentralizedController(env, factory.CreateDecentralizedAgents(env,
                                                                                       AttackNearestEnemy(),
                                                                                       AttackNearestEnemy()),
                                                        coordinator=coordinator.IdentityCoordinator(env),
                                                        plan_length=const.PLAN_LENGTH)
    return obj


def test_attack_nearest_coordinated(env):
    obj = factory.CreateDecentralizedController(env, factory.CreateDecentralizedAgents(env,
                                                                                       AttackNearestEnemy(),
                                                                                       AttackNearestEnemy()),
                                                coordinator=coordinator.SimGreedyCoordinator(env),
                                                plan_length=const.PLAN_LENGTH)
    return obj


def test_approx_best_action(env):
    obj = factory.CreateDecentralizedController(env, factory.CreateDecentralizedAgents(env,
                                                                                       ApproxBestAction,
                                                                                       AttackNearestEnemy(),
                                                                                       True, False))

    return obj


def test_approx_best_action_coordinated(env):
    obj = factory.CreateDecentralizedController(env, factory.CreateDecentralizedAgents(env,
                                                                                       ApproxBestAction,
                                                                                       ApproxBestAction,
                                                                                       True),
                                                coordinator=coordinator.SimGreedyCoordinator(env),
                                                plan_length=const.PLAN_LENGTH)
    return obj
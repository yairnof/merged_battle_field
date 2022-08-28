import copy

from agents import DecisionMaker
import networkx as nx
import numpy as np
import battle_field_ulits as utils
import random
import matplotlib.pyplot as plt
import constants as const
from ast import literal_eval

class RandomDecisionMaker:
    def __init__(self, action_space):
        self.space = action_space

    def get_action(self, observation):

        # TODO this assumes that the action space is a gym space with the `sample` funcition
        if isinstance(self.space, dict):
            return {agent: self.space[agent].sample() for agent in self.space.keys()}
        else:
            return self.space.sample()

    # Random plan - the result of a random sequence of get_action calls
    def get_plan(self, observation, plan_length):
        plan = [self.get_action(None) for _ in range(0, plan_length)]  # Blind action choice
        return plan


class Do_action_DM:
    def __init__(self, action_space, action=6):
        self.action_space = action_space
        self.steps = 0
        self.action = action

    def get_action(self, observation):
        self.steps += 1

        if (self.steps % 3 == 0):
            return self.action
        elif (self.steps % 3 == 1):
            return self.action_space.sample()
        else:
            return 2


class Stay_DM:
    def __init__(self, action_space, action):
        self.action_space = action_space
        self.steps = 0
        self.action = action

    def get_action(self, observation):
        self.steps += 1
        if (self.steps < 9):
            return self.action
        else:
            return 6


# A decision maker with a pre-defined joint plan, for simulation
class SimDecisionMaker(DecisionMaker):
    def __init__(self, joint_plan):
        self.update_plan(joint_plan)

    def get_action(self, observation):
        self.current_action_index += 1
        if self.current_action_index >= self.min_plan_length:
            return None
        return {agent_id: plan[self.current_action_index] for (agent_id, plan) in self.joint_plan.items()}

    def update_plan(self, joint_plan):
        self.joint_plan = joint_plan
        self.current_action_index = -1
        self.min_plan_length = min([len(plan) for plan in joint_plan.values()])


class AttackNearestEnemy(DecisionMaker):
    def __init__(self, env, agent_id):
        self.env = env
        self.agent_id = agent_id
        self.my_color = self.agent_id.split('_')[0]
        self.opponent_color = 'blue' if self.my_color == 'red' else 'red'
        self.iteration = 0
        # self.current_exploration_action = -1
        self.visit_count = {}

    # This decision maker does not return a single action
    def get_action(self, observation, return_agent_id=False):
        return self.get_plan(observation, 1, return_agent_id)[0]

    # Create a plan to reach the nearest enemy in the shortest path and attack him
    def get_plan(self, observation, plan_length, return_agent_id=False):
        curr_agent_pos = str(utils.agent_pos_from_its_obs(observation).tolist())
        if curr_agent_pos in self.visit_count:
            self.visit_count[curr_agent_pos] += 1
        else:
            self.visit_count[curr_agent_pos] = 1


        self.iteration += 1

        # enemy_list = utils.seen_agent_ids(observation, self.opponent_color)
        # enemy_list = list(set(enemy_list))  # Sometimes the same agent appears twice
        #
        # if len(enemy_list) == 0:
        #     # self.current_exploration_action = (self.current_exploration_action + 1) % const.MAX_MOVE_ACTION_IDX
        #     neighbor_cells = utils.neighbor_cells(observation)
        #     neighbor_visits = {str(cell): self.visit_count[str(cell)] for cell in neighbor_cells if str(cell) in self.visit_count}
        #     if len(neighbor_visits) < const.MAX_MOVE_ACTION_IDX:
        #         min_neighbor = random.choice([cell for cell in neighbor_cells if str(cell) not in neighbor_visits])
        #     else:
        #         min_neighbor  = min(neighbor_visits, key=neighbor_visits.get)
        #         min_neighbor = literal_eval(min_neighbor)
        #     neighbor_diff = min_neighbor - utils.agent_pos_from_its_obs(observation)
        #     return [utils.diff_to_action_num(neighbor_diff.tolist())]

            #return [self.current_exploration_action]

        agent_map = utils.map_around_agent(observation)
        grid_graph = self.build_grid_graph(agent_map, const.OBS_SIZE)
        enemies = utils.enemies_around_agent(observation)

        # Global state - too slow
        # agent_map = utils.state_grid(self.env.env.state())
        # grid_graph = self.build_grid_graph(agent_map, const.MAP_SIZE)
        # enemies = utils.state_enemies(self.env.env.state(), self.my_color)

        enemies_pos = np.argwhere(enemies == 1)
        if enemies_pos.size == 0:
            if return_agent_id:
                return self.agent_id, [random.randint(const.MIN_ACTION_IDX, const.MAX_MOVE_ACTION_IDX)]
            else:
                return [random.randint(const.MIN_ACTION_IDX, const.MAX_MOVE_ACTION_IDX)]

        enemy_set = set([f"{x}:{y}" for [x, y] in enemies_pos.tolist()])
        agent_pos = [const.OBS_SIZE // 2,
                     const.OBS_SIZE // 2]  # utils.agent_pos_from_its_obs(observation).astype(int) if it was absolute position

        agent_pos_str = self.list_pos_to_str(agent_pos)
        try:
            # plt.imshow(agent_map)
            # plt.show()
            multi_source_length, multi_source_path = nx.multi_source_dijkstra(grid_graph, enemy_set, agent_pos_str)
            # node_colors = ["blue" if n in multi_source_path else "red" for n in grid_graph.nodes()]
            # nx.draw(grid_graph, nx.get_node_attributes(grid_graph, 'pos'), node_color=node_colors, with_labels=True, node_size=10)
            # plt.show()
        except Exception as e:
            if return_agent_id:
                return self.agent_id, [random.randint(const.MIN_ACTION_IDX, const.MAX_ACTION_IDX)]
            else:
                return [random.randint(const.MIN_ACTION_IDX, const.MAX_ACTION_IDX)]

        # Instead of reaching the cell of the enemy - Attack it
        path_to_enemy = multi_source_path[::-1]
        route = path_to_enemy[0:-1]
        diff = self.str_pos_diff(path_to_enemy[-1], path_to_enemy[-2])
        attack_action = utils.enemy_dir_to_attack_action(diff)
        plan = utils.route_to_actions(self.str_route_to_list(route)) + [attack_action]

        if return_agent_id:
            return self.agent_id, plan
        else:
            return plan

    # Build a graph from a grid which is a binary matrix where 0 represent a free cell and any other value is a blocked cell
    def build_grid_graph(self, map_matrix, size):
        grid_graph = nx.Graph()
        poses = {}
        for i in range(size):
            for j in range(size):
                if map_matrix[i][j] == 0:
                    self.connect_neighbors(i, j, [(i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1)], map_matrix,
                                           grid_graph, size, w=1)
                    self.connect_neighbors(i, j, [(i + 1, j + 1), (i + 1, j - 1), (i - 1, j + 1), (i - 1, j - 1)],
                                           map_matrix, grid_graph, size, w=np.sqrt(2))
        return grid_graph

    def connect_neighbors(self, i, j, neighbors, matrix, graph, size, w):
        for (x, y) in neighbors:
            if x in range(size) and y in range(size) and matrix[x][y] == 0:
                graph.add_edge(f'{i}:{j}', f'{x}:{y}', weight=w)
                graph.nodes[f'{i}:{j}']['pos'] = (j, i)
                graph.nodes[f'{x}:{y}']['pos'] = (y, x)

    def str_pos_to_list(self, str_pos):
        return [int(i) for i in str_pos.split(":")]

    def list_pos_to_str(self, list_pos):
        return f"{list_pos[0]}:{list_pos[1]}"

    def str_pos_diff(self, str_pos_1, str_pos_2):
        array_diff = np.array(self.str_pos_to_list(str_pos_1)) - np.array(self.str_pos_to_list(str_pos_2))
        return array_diff.tolist()

    def str_route_to_list(self, str_route):
        return [self.str_pos_to_list(str_pos) for str_pos in str_route]

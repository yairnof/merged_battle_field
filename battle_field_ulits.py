import numpy as np
import constants as const

# Each tuple contains the action number, a readable name, x difference and y difference
action_tuples = [(0, '2-Up', 0, -2), (1, 'Up-Left', -1, -1), (2, '1-Up', 0, -1), (3, 'Up-Right', 1, -1),
                 (4, '2-Left', -2, 0), (5, '1-Left', -1, 0), (6, 'Do-Nothing', 0, 0), (7, '1-Right', 1, 0),
                 (8, '2-Right', 2, 0), (9, 'Down-Left', -1, 1), (10, '1-Down', 0, 1), (11, 'Down-Right', 1, 1),
                 (12, '2-Down', 0, 2),
                 (13, 'Attack-Up-Left', 0, 0), (14, 'Attack-Up', 0, 0), (15, 'Attack-Up-Right', 0, 0),
                 (16, 'Attack-Left', 0, 0), (17, 'Attack-Right', 0, 0), (18, 'Attack-Down-Left', 0, 0),
                 (19, 'Attack-Down', 0, 0), (20, 'Attack-Down-Right', 0, 0)]

# What attack action to take in each case of a neighbor enemy
attack_dir = [(13, 'Attack-Up-Left', -1, -1), (14, 'Attack-Up', 0, -1), (15, 'Attack-Up-Right', 1, -1),
              (16, 'Attack-Left', -1, 0), (17, 'Attack-Right', 1, 0), (18, 'Attack-Down-Left', -1, 1),
              (19, 'Attack-Down', 0, 1), (20, 'Attack-Down-Right', 1, 1)]


# Name the properties of a cell for a single agent, from observation - with minimap mode
def obs_features_for_agent_at(obs, agent_id, i, j):
    ob = obs[agent_id]
    if len(ob[0,0])== 9: # case only minimap on
        return {'is_blocked': obs[agent_id][i, j][0], 'my_team_presence': obs[agent_id][i, j][1],
                'my_team_hp': obs[agent_id][i, j][2], 'my_team_minimap': obs[agent_id][i, j][3],
                'other_team_presence': obs[agent_id][i, j][4], 'other_team_hp': obs[agent_id][i, j][5],
                'other_team_minimap': obs[agent_id][i, j][6],
                'agent_position': np.round(obs[agent_id][0, 0, 7:9] * const.MAP_SIZE)}
    elif len(ob[0,0])== 5: # case no minimap no extra
        return {'is_blocked': obs[agent_id][i, j][0], 'my_team_presence': obs[agent_id][i, j][1],
                'my_team_hp': obs[agent_id][i, j][2],
                'other_team_presence': obs[agent_id][i, j][3], 'other_team_hp': obs[agent_id][i, j][4]}
    elif len(ob[0, 0]) == 41: # case minimap and extra on
        return {'is_blocked': obs[agent_id][i, j][0], 'my_team_presence': obs[agent_id][i, j][1],
                'my_team_hp': obs[agent_id][i, j][2], 'my_team_minimap': obs[agent_id][i, j][3],
                'other_team_presence': obs[agent_id][i, j][4], 'other_team_hp': obs[agent_id][i, j][5],
                'other_team_minimap': obs[agent_id][i, j][6], 'binary_agent_id': obs[agent_id][i, j][7:17],
                'one_hot_action': obs[agent_id][i, j][17:38], 'last_reward': obs[agent_id][i, j][38],
                'agent_position': np.round(obs[agent_id][0, 0, 39:41] * const.MAP_SIZE)}
    else:
        return {'is_blocked': obs[agent_id][i, j][0], 'my_team_presence': obs[agent_id][i, j][1],
                'my_team_hp': obs[agent_id][i, j][2],
                'other_team_presence': obs[agent_id][i, j][3], 'other_team_hp': obs[agent_id][i, j][4],
                'binary_agent_id': obs[agent_id][i, j][5:15],
                'one_hot_action': obs[agent_id][i, j][15:36], 'last_reward': obs[agent_id][i, j][36]}


def seen_agent_ids(obs, opponent_color):
    if len(obs[0, 0]) == 9 or len(obs[0, 0]) == 5:  # case only minimap on or no minimap no extra
        assert True
    elif len(obs[0, 0]) == 41:  # case minimap and extra on
        id_range = range(7, 17)
        enemy_idx = 4
    else:
        id_range = range(5, 15)
        enemy_idx = 3

    agent_ids = []
    for i in range(const.OBS_SIZE):
        for j in range(const.OBS_SIZE):
            if obs[i, j][enemy_idx]==1:
                digits_arr = obs[i, j][id_range].astype(int)
                char_arr = [str(i) for i in digits_arr.tolist()]
                char_arr.reverse()
                agent_id = int(''.join(char_arr), 2)
                agent_name = opponent_color + '_' + str(agent_id)
                agent_ids.append(agent_name)

    return agent_ids


# Get action's number from its name
def action_num_to_str(action_number):
    action_str = [name for (num, name, x, y) in action_tuples if num == action_number]
    return action_str[0]


# Get action's name from its number
def action_str_to_num(action_str):
    action_num = [num for (num, name, x, y) in action_tuples if name == action_str]
    return action_num[0]


# Get action's x and y differences from its number
def action_num_to_diff(action_num):
    pos = [(x, y) for (num, name, x, y) in action_tuples if num == action_num]
    return pos[0]


# Get action num from desired diff
def diff_to_action_num(diff):
    action_num = [num for (num, name, x, y) in action_tuples if [x, y] == diff]
    return action_num[0]


# Use of attack_dir
def enemy_dir_to_attack_action(enemy_dir):
    actions = [num for (num, name, x, y) in attack_dir if [y, x] == enemy_dir]
    return actions[0]


# Translate desired agent positions to corresponding actions
def route_to_actions(route):
    route = [[y, x] for [x, y] in route]
    if len(route) == 1:
        return []
    diff_seq = [(np.array(j)-np.array(i)).tolist() for i, j in zip(route[:-1], route[1:])]
    return [diff_to_action_num(d) for d in diff_seq]


# Build an array as list of lists from obs_features_for_agent_at
def obs_features_for_agent(obs, agent_id):
    features_for_agent = []
    for i in range(0, 13):
        row = []
        for j in range(0, 13):
            row.append(obs_features_for_agent_at(obs, agent_id, i, j))
        features_for_agent.append(row)

    return features_for_agent


# Build a dictionary of list of lists using obs_features_for_agent
def obs_features(obs):
    agent_ids = obs.keys()
    return {agent_id: obs_features_for_agent(obs, agent_id) for agent_id in agent_ids}


# Create a list of observation features
def obs_seq_features(observations):
    return [obs_features(obs) for obs in observations]


# Get a single agent's position from an observation
def agent_pos(obs, agent_id):
    return obs_features_for_agent(obs, agent_id)[0][0]['agent_position'].tolist()


# Get all agents' positions from an observation
def all_agent_pos(obs):
    agent_ids = obs.keys()
    return {agent_id: agent_pos(obs, agent_id) for agent_id in agent_ids}


# Get a list of single agent positions from a list of observations
def agent_pos_seq(observations, agent_id):
    return [agent_pos(obs, agent_id) for obs in observations]


# Get a list of all agents' positions from a list of observations
def all_agents_pos_seq(observations):
    return [all_agent_pos(obs) for obs in observations]


# Position only by advancing initial observation using the plan
def est_agent_pos_seq(initial_obs, agent_id, plan):
    # e_pos = [agent_pos(initial_obs, agent_id)] # Taking too long to compute

    e_pos = [np.round(agent_pos_from_its_obs(initial_obs[agent_id])).tolist()]

    for i in range(1, len(plan)):
        pos = action_num_to_diff(plan[i])
        e_pos.append([sum(x) for x in zip(e_pos[i - 1], pos)])

    return e_pos


# Positions for all agents only by advancing initial observation using the joint plan
def all_est_agents_pos_seq(initial_obs, joint_plan):
    agent_ids = joint_plan.keys()
    return {agent_id: est_agent_pos_seq(initial_obs, agent_id, joint_plan[agent_id]) for agent_id in agent_ids}


def map_around_agent(agent_obs):
    agent_map = np.empty([const.OBS_SIZE, const.OBS_SIZE])
    for i in range(const.OBS_SIZE):
        for j in range(const.OBS_SIZE):
            agent_map[i][j] = agent_obs[i, j][0]
    return agent_map


def enemies_around_agent(agent_obs):
    agent_enemies = np.empty([const.OBS_SIZE, const.OBS_SIZE])
    for i in range(const.OBS_SIZE):
        for j in range(const.OBS_SIZE):
            agent_enemies[i][j] = agent_obs[i, j][4]
    return agent_enemies


def team_around_agent(agent_obs):
    agent_enemies = np.empty([const.OBS_SIZE, const.OBS_SIZE])
    for i in range(const.OBS_SIZE):
        for j in range(const.OBS_SIZE):
            agent_enemies[i][j] = agent_obs[i, j][1]
    return agent_enemies


# Works only with minimap=True and extra_features=True
def agent_pos_from_its_obs(agent_obs):
    return np.round(agent_obs[0, 0, 39:41] * const.MAP_SIZE)

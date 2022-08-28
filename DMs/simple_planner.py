import random

from agents import DecisionMaker, Agent
from control import CentralizedController
from copy import deepcopy
import constants as const
import factory
import performance
import battle_field_ulits as utils
from DMs.simple_DMs import SimDecisionMaker


class Simple_DM(DecisionMaker):
    def __init__(self, action_space, health_th=0.5 , red_team=False):
        self.space = action_space
        self.healt = None
        self.healt_th = health_th
        self.walls = []
        self.my_team = []
        self.op_team = []
        self.is_red_team = red_team

    def set_state(self, obs):
        self.walls = []
        self.my_team = []
        self.op_team = []
        self.healt = obs[6,6,2]
        for y in range (13):
            for x in range (13):
                if x==6 and y==6:
                    continue
                if obs[y,x,0]==1:
                    self.walls.append((x-6,y-6))
                elif obs[y,x, 1] == 1:
                    self.my_team.append(((x-6,y-6),obs[y,x,2]))
                elif obs[y,x,3] == 1:
                    self.op_team.append(((x-6,y-6), obs[y,x, 4]))

    def defensive_move(self):
        if self.is_red_team:
            return 4
        else: return 8
        #todo set defensive better

    def search_opponent(self):
        if self.is_red_team:
            if ((1, 0) in self.walls) or ((2, 0) in self.walls):
                return 0
            else:
                return random.choice([3, 8, 7, 11])
        else:
            if ((-1,0) in self.walls) or ((-2,0) in  self.walls):
                return 0
            else:
                return random.choice([1,4,5,9])

    def attack_range(self):
        if len(self.op_team)==0 : return None
        for (x,y) in self.op_team:
            if abs(x[0])<2 and abs(x[1])<2 :
                return (x)
        return None

    def attack(self,pos):
        "translate position to related attack action"
        i = 4 + pos[0] + (3*pos[1])
        if i>4 : i-=1
        return (13 + i)

    def go_to(self,pos):
        "translate position to related  action"
        i = 6 + pos[0] + (4*pos[1])
        return i

    def find_closest(self):
        min = 13
        for x,_ in self.op_team:
            dist = abs(x[0])+ abs(x[1])
            if dist<min :
                close_pos = x
                min = dist
        return close_pos

    def check_wall(self,pos):
        if (pos in self.walls):
            return True
        else: return False

    def check_my_team(self,pos):
        if (pos in [x[0] for x in self.walls]):
            return True
        else: return False

    def act_to_pos(self,act):
        pos = [(0,-2),(-1,-1),(0,-1),(1,-1),
               (-2,0),(-1,0),(0,0),(1,0),(2,0),
               (-1,1),(0,1),(1,1),(0,2)]
        return pos[act]


    def chase_closest(self):
        close = self.find_closest()
        if close[1]>2:
            if self.check_my_team(self.act_to_pos(12)):
                return 10
            else: return 12
        if close[1]<-2:
            if self.check_my_team(self.act_to_pos(0)):
                return 2
            return 0
        if close[0]<-2:
            if self.check_wall((-2,0)) or self.check_wall((-1,0)):
                if self.check_my_team(self.act_to_pos(0)):
                    return 2
                return 0
            else: return 4
        if close[0]>2:
            if self.check_wall((2,0)) or self.check_wall((1,0)):
                if self.check_my_team(self.act_to_pos(12)):
                    return 10
                return 12
            else:
                if self.check_my_team(self.act_to_pos(8)):
                    return 7
                return 8
        #
        if close[0]<0:
            x = -1
        elif close[0]==0:
            x = 0
        else: x = 1
        #
        if close[1]<0:
            y = -1
        elif close[1]==0:
            y = 0
        else: y = 1
        return self.go_to((x,y))


    def get_action(self, observation):
        if random.uniform(0,1)<0.01:
            return self.space.sample()
        self.set_state(observation)
        if self.healt<self.healt_th and len(self.op_team)>0:
            return self.defensive_move()
        if len(self.op_team)==0:
            return self.search_opponent()
        atk = self.attack_range()
        if (atk!= None):
            return self.attack(atk)
        else:
            return self.chase_closest()


class ApproxBestAction(DecisionMaker):
    def __init__(self, env, agent_id):
        self.original_env = env
        self.sim_env = deepcopy(env)
        self.agent_id = agent_id
        self.my_color = self.agent_id.split('_')[0]
        self.opponent_color = 'blue' if self.my_color == 'red' else 'red'
        self.spaces = env.action_spaces
        self.sim_controller = CentralizedController(self.sim_env, Agent(SimDecisionMaker(joint_plan={'dummy': [0]})))
        self.iteration = 0
        self.current_exploration_action = -1


    def get_action(self, observation, return_agent_id=False):
        # if random.random() < 0.99:
        # return self.spaces[self.agent_id].sample()
        # if int(self.agent_id.split('_')[1]) > 12:
        #     return utils.action_str_to_num('1-Down')
        self.iteration += 1

        enemy_list = utils.seen_agent_ids(observation, self.opponent_color)
        enemy_list = list(set(enemy_list)) # Sometimes the same agent appears twice

        if len(enemy_list) == 0:
            return self.agent_id, random.randint(const.MIN_ACTION_IDX, const.MAX_MOVE_ACTION_IDX) if return_agent_id else random.randint(const.MIN_ACTION_IDX, const.MAX_MOVE_ACTION_IDX)
            # return self.agent_id, self.current_exploration_action if return_agent_id else self.current_exploration_action

        best_action = const.MIN_ACTION_IDX
        best_value = const.REWARD_SUM_LB
        for action in range(const.MIN_ACTION_IDX, const.MAX_ACTION_IDX):
            value = self.simulate_action(action, enemy_list)
            if value > best_value:
                best_value = value
                best_action = action

        return self.agent_id, best_action if return_agent_id else best_action

    def simulate_action(self, action, enemy_list):
        opponent_actions = {agent: [self.spaces[agent].sample()] for agent in enemy_list if self.opponent_color in agent}
        opponent_actions[self.agent_id] = [action]
        self.sim_controller.central_agent.decision_maker.update_plan(opponent_actions)
        plan_length = min([len(plan) for (agent, plan) in opponent_actions.items()])
        self.sim_controller.run(render=False, max_iteration=plan_length)

        self.sim_env = deepcopy(self.original_env)  # Rewind simulated environment
        reward_sum = performance.colored_total_rewards(self.sim_controller.total_rewards)
        return reward_sum[self.my_color] - reward_sum[self.opponent_color]
        # best_opponent_reward_sum = self.best_opponent_response(action, enemy_list)
        # return -best_opponent_reward_sum


    def best_opponent_response(self, action, enemy_list):
        pass
        ## This search is taking too long-  Stochastic hill climbing
        # current_opponent_actions = {agent: [self.spaces[agent].sample()] for agent in enemy_list if self.opponent_color in agent}
        # best_opponent_reward_sum = const.REWARD_SUM_LB
        #
        # for i in range(const.OPPONENT_ITERATIONS):
        #     selected_changes = random.sample(list(current_opponent_actions.keys()), min(const.NEIGHBORHOOD_SIZE, len(enemy_list)))
        #     opponent_actions_candidate = {agent: [self.spaces[agent].sample()] if agent in selected_changes else current_opponent_actions[agent] for agent in current_opponent_actions.keys()}
        #     checked_actions = opponent_actions_candidate.copy()
        #     checked_actions[self.agent_id] = [action]
        #     total_rewards, sim_obs_seq = factory.CreateSimulationController(self.sim_env, checked_actions)
        #     self.sim_controller.central_agent.decision_maker.update_plan(checked_actions)
        #     plan_length = min([len(plan) for (agent, plan) in checked_actions.items()])
        #     self.sim_controller.run(render=False, max_iteration=plan_length)
        #     reward_sum = performance.colored_total_rewards(self.sim_controller.total_rewards)
        #     opponent_reward_sum = reward_sum[self.opponent_color]
        #     if opponent_reward_sum > best_opponent_reward_sum:
        #         best_opponent_reward_sum = opponent_reward_sum
        #         current_opponent_actions = opponent_actions_candidate.copy()
        #     self.sim_env = deepcopy(self.original_env)  # Rewind simulated environment
        #
        # return best_opponent_reward_sum

    def get_plan(self, observation, plan_length, return_agent_id=False):
        return self.get_action(observation, return_agent_id)



class Centralized_programmed_DM(DecisionMaker):
    def __init__(self, env, my_color):
        self.env = env
        self.my_color = my_color
        self.opponent_color = 'blue' if self.my_color == 'red' else 'red'
        self.spaces = env.action_spaces

    def get_action(self, observation):
        # enemy_agents = [agent for agent in self.env.get_env_agents() if self.opponent_color in agent]
        my_random_actions = {agent: [self.spaces[agent].sample()] for agent in self.env.get_env_agents() if
                              self.my_color in agent}
        return my_random_actions


class Centralized_Search_DM(DecisionMaker):
    def __init__(self, env, my_color):
        self.original_env = env
        self.sim_env = deepcopy(env)
        self.my_color = my_color
        self.opponent_color = 'blue' if self.my_color == 'red' else 'red'
        self.spaces = env.action_spaces
        self.sim_controller = CentralizedController(self.sim_env,
                                                    Agent(SimDecisionMaker(joint_plan={'dummy': [0]})))

    def get_action(self, observation):
        # if random.random() < 0.99:
        # return self.spaces[self.agent_id].sample()

        my_current_actions = {agent: [self.spaces[agent].sample()] for agent in self.sim_env.get_env_agents() if
                              self.my_color in agent}
        opponent_actions = {agent: [self.spaces[agent].sample()] for agent in self.sim_env.get_env_agents() if
                            self.opponent_color in agent}
        best_reward_diff = const.REWARD_SUM_LB

        for i in range(const.MY_ITERATIONS):
            selected_changes = random.sample(list(my_current_actions.keys()), const.NEIGHBORHOOD_SIZE)
            actions_candidate = {
                agent: [self.spaces[agent].sample()] if agent in selected_changes else my_current_actions[agent] for
                agent in my_current_actions.keys()}
            checked_actions = actions_candidate.copy()
            checked_actions = {**checked_actions, **opponent_actions}

            self.sim_controller.central_agent.decision_maker.update_plan(checked_actions)
            plan_length = min([len(plan) for (agent, plan) in checked_actions.items()])
            self.sim_controller.run(render=False, max_iteration=plan_length)
            reward_sum = performance.colored_total_rewards(self.sim_controller.total_rewards)
            reward_diff = reward_sum[self.my_color] - reward_sum[self.opponent_color]
            if reward_diff > best_reward_diff:
                best_reward_diff = reward_diff
                my_current_actions = actions_candidate.copy()
            self.sim_env = deepcopy(self.original_env)  # Rewind simulated environment

        return my_current_actions

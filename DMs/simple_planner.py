import random

from agents import DecisionMaker
from copy import deepcopy
import constants as const
import factory
import performance

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

    def get_action(self, observation):
        best_action = const.MIN_ACTION_IDX
        best_value = const.REWARD_SUM_LB
        for action in range(const.MIN_ACTION_IDX, const.MAX_ACTION_IDX):
            value = self.simulate_action(action)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def simulate_action(self, action):
        best_opponent_reward_sum = self.best_opponent_response(action)
        return -best_opponent_reward_sum

    # Stochastic hill climbing
    def best_opponent_response(self, action):
        current_opponent_actions = {agent: [self.spaces[agent].sample()] for agent in self.spaces.keys() if self.opponent_color in agent}
        best_opponent_reward_sum = const.REWARD_SUM_LB

        for i in range(const.OPPONENT_ITERATIONS):
            selected_changes = random.sample(list(current_opponent_actions.keys()), const.NEIGHBORHOOD_SIZE)
            opponent_actions_candidate = {agent: [self.spaces[agent].sample()] if agent in selected_changes else current_opponent_actions[agent] for agent in current_opponent_actions.keys()}
            checked_actions = opponent_actions_candidate.copy() # deepcopy(opponent_actions_candidate)
            checked_actions[self.agent_id] = [action]
            total_rewards, sim_obs_seq = factory.CreateSimulationController(self.sim_env, checked_actions)
            reward_sum = performance.colored_total_rewards(total_rewards)
            opponent_reward_sum = reward_sum[self.opponent_color]
            if opponent_reward_sum > best_opponent_reward_sum:
                best_opponent_reward_sum = opponent_reward_sum
                current_opponent_actions = opponent_actions_candidate.copy()
            # self.sim_env = deepcopy(self.original_env)  # Rewind simulated environment

        return best_opponent_reward_sum






from abc import ABC, abstractmethod
from copy import deepcopy
import factory
import performance
import battle_field_ulits as utils


# The abstract coordinator class with the abstract method approve_joint_plan
class coordinator(ABC):
    def __init__(self, env):
        self.OriginalEnv = env  # Original environment is saved to enable several resets to the simulated environment
        self.SimEnv = deepcopy(
            env)  # Create a copy of the environment to enable environment steps without changing the real environment

    @abstractmethod
    def approve_joint_plan(self,joint_plan, obs=None):  # coordinate_color is the color which makes decisions. The other color is simulated
        pass


# for testing: A controller that allow all plans - it returns for each agent the first action of its plan,
# as the next action for a controller
class IdentityCoordinator(coordinator):
    def __init__(self, env):
        super().__init__(env)

    def approve_joint_plan(self, joint_plan, obs=None):
        return {agent: plan[0] for (agent, plan) in joint_plan.items()} # Take the first action of each joint plan, as is


# A coordinator which consider which plans to allow by simulating it in the environment.
# The greedy method in this case try and check in each iteration what is the best plan, given the previous plans
# This coordinator can be used for both sizes
class SimGreedyCoordinator(coordinator):
    def __init__(self, env):
        super().__init__(env)

    # This is the greedy vs. greedy coordination
    # def approve_joint_plan(self, joint_plan, obs=None):
    #     approved_blue_plans = self.greedy_coordination(joint_plan, 'blue', obs)
    #     approved_red_plans = self.greedy_coordination(joint_plan, 'red', obs)
    #     approved_plans = {**approved_red_plans, **approved_blue_plans}
    #     return {agent: plan[0] for (agent, plan) in approved_plans.items()}

    # This is 'no coordination' vs. greedy coordination
    def approve_joint_plan(self, joint_plan, obs=None):
        approved_blue_plans = {agent: plan[0] for (agent, plan) in joint_plan.items() if 'blue' in agent}
        approved_red_plans = self.greedy_coordination(joint_plan, 'red', obs)
        return {**approved_red_plans, **approved_blue_plans}

    # This is 'no coordination' vs. 'no coordination'
    # def approve_joint_plan(self, joint_plan, obs=None):
    #     approved_blue_plans = {agent: plan[0] for (agent, plan) in joint_plan.items() if 'blue' in agent}
    #     approved_red_plans = {agent: plan[0] for (agent, plan) in joint_plan.items() if 'blue' in agent}
    #     return {**approved_red_plans, **approved_blue_plans}

    def greedy_coordination(self, joint_plan, coordinate_color, obs=None, use_simulation=False):
        # opponent_color = 'blue' if coordinate_color == 'red' else 'red' # Suppose we don't know opponent plans
        # opponent_plans = {agent: plan for (agent, plan) in joint_plan.items() if opponent_color in agent} # Suppose we don't know opponent plans
        coordination_plans = {agent: plan for (agent, plan) in joint_plan.items() if coordinate_color in agent}
        checked_plans = {}

        for (agent, plan) in coordination_plans.items():  # Consider adding each plan
            checked_plans[agent] = plan

            if use_simulation:
                # Simulate adding the current plan
                total_rewards, sim_obs_seq = factory.CreateSimulationController(self.SimEnv, checked_plans)
                initial_obs = sim_obs_seq[0]
            else:
                initial_obs = obs
                sim_obs_seq = None

            # Intended positions, before the environment prevents collisions
            estimated_poses = utils.all_est_agents_pos_seq(initial_obs, checked_plans)

            for (prev_agent, prev_plan) in checked_plans.items():  # Checking hard constraints vs. previous taken plans
                if prev_agent == agent: # or opponent_color in prev_agent or # Suppose we don't know opponent plans
                    continue

                # If hard constraints are violated, current plan is not taken
                if performance.forbidden_plans(sim_obs_seq, prev_plan, prev_agent, plan, agent, estimated_poses):
                    checked_plans[agent] = self.default_action(checked_plans)

        # Dropping opponent plans for return
        return {agent: plan[0] for (agent, plan) in checked_plans.items() if coordinate_color in agent}

    # Create a plan with default action for a length of the minimum given joint plan
    def default_action(self, joint_plan):
        plan_length = min([len(plan) for (agent, plan) in joint_plan.items()])
        return [utils.action_str_to_num('Do-Nothing') for i in range(plan_length)]






from .controller import Controller
import numpy as np


"""Abstract parent class for centralized controller 
"""
class CentralizedController(Controller):

    def __init__(self, env, central_agent):
        # initialize super class
        super().__init__(env)

        self.central_agent = central_agent
        self.observations = []
        self.step = 1

    def get_joint_action(self, observation):
        """Returns the joint actions of all the agents

        Args:
            observation ([dict]): The agents observations

        Returns:
            dict: dict of all the actions
        """

        observation = {agent_id: self.central_agent.get_observation(obs)
                       for agent_id, obs in observation.items()}
        # save observations (step,observations of this step)
        self.observations.append(observation)
        self.step += 1

        return self.central_agent.get_decision_maker().get_action(observation)

    # temp implementation
    def decode_state(self, obs):
        return obs

    def decode_action(self, action, num_agents):
        """Decodes the action from the model to RL env friendly format

        Args:
            action (int): The action from the model
            num_agents (int): number of agents

        Returns:
            list: list of individual actions
        """
        out = {}
        for ind in range(num_agents):
            out.append(action % num_actions)
            action = action // num_actions
        return list(reversed(out))
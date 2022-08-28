from control import Controller
from joblib import Parallel, delayed
import constants as const


class MultiDecisionAgentsController(Controller):

    def __init__(self, env, decision_agents):
        # initialize super class
        super().__init__(env)

        self.decision_agents = decision_agents
        self.observations = []
        self.step = 1

    def get_joint_action(self, observation):
        """Returns the joint actions of all the agents

        Args:
            observation ([dict]): The agents observations

        Returns:
            dict: dict of all the actions
        """

        observation = {agent_id: obs for agent_id, obs in observation.items()}

        # save observations (step,observations of this step)

        self.observations.append(observation)
        self.step += 1

        joint_action = {}

        if const.PARALLEL:
            delayed_funcs = [delayed(decision_agent.get_decision_maker().get_action)(observation)
                             for decision_agent in self.decision_agents]

            list_action_dict = Parallel(n_jobs=len(delayed_funcs))(delayed_funcs)

            for d in list_action_dict:
                joint_action.update(d)

        else:
            for decision_agent in self.decision_agents:
                actions = decision_agent.get_decision_maker().get_action(observation)
                joint_action.update(actions)

        return joint_action



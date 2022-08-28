from abc import ABC, abstractmethod


class Agent:

    def __init__(self, decision_maker, sensor_function=None, message_filter=None):
        self.decision_maker = decision_maker
        self.sensor_function = sensor_function or (lambda x: x)  # default to identity function
        self.message_filter = message_filter

    def get_decision_maker(self):
        return self.decision_maker

    def get_observation(self, state):
        return self.sensor_function(state)


class DecisionMaker(ABC):
    """
    An abstract class for choosing an action, part of an agent.
    (An agent can have one or several of these)
    """

    @abstractmethod
    def get_action(self, observation, return_agent_id=False):
        pass

    # A decision maker can calculate and return a whole deterministic plan
    def get_plan(self, observation, plan_length, return_agent_id=False):
        return [self.get_action(observation)]  # The default plan is the one step plan given by get_action

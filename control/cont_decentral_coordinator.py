from .controller import Controller
import constants as const
from joblib import Parallel, delayed, parallel_backend

class DecentralizedControllerCoordinator(Controller):

    def __init__(self, env, agents, coordinator=None, plan_length=0):
        # initialize super class
        super().__init__(env)
        self.coordinator = coordinator  # If a coordinator is given, it will be used to approve each plan
        self.plan_length = plan_length  # A coordinator can be used with decision makers that return a plan with plan_length, not only an action

        # safely accept agents a dict or as a list of agents matching the agent_ids list order
        assert len(agents) == len(self.agent_ids)
        if isinstance(agents, dict):
            assert all(agent in self.agent_ids for agent in self.agent_ids)
            self.agents = agents
        elif isinstance(agents, list):
            self.agents = {id_: agent for id_, agent in zip(self.agent_ids, agents)}

    def get_joint_action(self, observation):
        """Returns the joint action

        Args:
            observation (dict): the current observatins

        Returns:
            dict: the actions for the agents
        """
        observation = {agent_id: self.agents[agent_id].get_observation(obs)
                       for agent_id, obs in observation.items()}

        joint_action = {}
        joint_plan = {}

        if const.PARALLEL:
            if self.coordinator is None:
                delayed_funcs = [delayed(self.agents[agent_name].get_decision_maker().get_action)(observation[agent_name], True)
                                 for agent_name in self.agent_ids if observation.get(agent_name) is not None]
            else:
                delayed_funcs = [
                    delayed(self.agents[agent_name].get_decision_maker().get_plan)(observation[agent_name], self.plan_length, True)
                    for agent_name in self.agent_ids if observation.get(agent_name) is not None]

            with parallel_backend(backend="threading"):
                list_tuple_actions = Parallel(n_jobs=-1, verbose=5)(delayed_funcs)



            if self.coordinator is None:
                joint_action = dict(list_tuple_actions)
            else:
                joint_plan = dict(list_tuple_actions)

            # for d in list_action_dict:
            #     joint_action.update(d)

        else:
            for agent_name in self.agent_ids:
                if observation.get(agent_name) is None:
                    continue
                if self.coordinator is not None:  # If there's a coordinator, the decision maker returns a plan
                    try:
                        plan = self.agents[agent_name].get_decision_maker().get_plan(observation[agent_name], self.plan_length)
                    except:
                         plan = self.agents[agent_name].get_decision_maker().get_action(observation[agent_name])
                    joint_plan[agent_name] = plan
                else:
                    action = self.agents[agent_name].get_decision_maker().get_action(observation[agent_name])
                    joint_action[agent_name] = action

        # The coordinator's approve_joint_action returns the next joint action, after considering the joint plan
        if self.coordinator is not None:
            joint_action = self.coordinator.approve_joint_plan(joint_plan, observation)

        print('Another one bites the dust')
        return joint_action

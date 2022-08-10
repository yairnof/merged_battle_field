#imports
from DMs.simple_planner import Simple_DM
import time

## Main
# from environments.env_wrapper import BattleFieldSingleEnv
from environments.env_wrapper import BattleFieldSingleEnv, CreateEnvironment

if __name__ == '__main__':
    env = CreateEnvironment()
    agent = "blue_11"
    action_space = env.action_spaces[agent]
    temp_env = BattleFieldSingleEnv(env, Simple_DM(action_space,0.5), Simple_DM(action_space,0.5,red_team=True), agent)

    obs = temp_env.reset()

    simple_dm = Simple_DM(temp_env.action_space)

    total_reward = 0
    for i in range(100):
        a = simple_dm.get_action(obs)
        obs,rew,done,_ = temp_env.step(a)
        if done:
            break
        temp_env.render()
        total_reward+=rew
        print(f"action: {a}, reward: {rew}, total rew: {total_reward}")
        time.sleep(0.3)

    print(f" total rew: {total_reward}")





    # mac_BF_env = CreateEnvironment()

    # CreateCentralizedController(mac_BF_env, CreateRandomAgent(mac_BF_env))

    # CreateDecentralizedController(mac_BF_env, CreateDecentralizedIdenticalAgents(mac_BF_env, RandomDecisionMaker))

    # CreateDecentralizedController(mac_BF_env, CreateDecentralizedAgents(mac_BF_env, Stay_DM , Stay_DM))

    # CreateDecentralizedController(mac_BF_env, CreateDecentralizedAgents(mac_BF_env, RandomDecisionMaker, RandomDecisionMaker))


    # GDM = GreedyDecisionMaker(mac_BF_env)

    # GDM.get_action()

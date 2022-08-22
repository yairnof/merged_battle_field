from warnings import filterwarnings
import factory
import tests
import performance
import cProfile

# Just to silent an harmless warning
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

mac_BF_env = factory.CreateEnvironment()

with cProfile.Profile() as pr:

    # tests.test_centralized_controller(mac_BF_env)

    # tests.test_decentralized_controller(mac_BF_env)

    # tests.test_sim_controller(mac_BF_env)

    # tests.test_coordinator(mac_BF_env)

    # tests.test_sim_coordinator(mac_BF_env)

    # tests.test_simple_dm(mac_BF_env)

    # tests.test_simple_dm_coordinated(mac_BF_env)

    # tests.test_attack_nearest(mac_BF_env)

    # tests.test_attack_nearest_coordinated(mac_BF_env)

    tests.test_approx_best_action(mac_BF_env)

    # tests.test_approx_best_action_coordinated(mac_BF_env)

pr.print_stats()

from warnings import filterwarnings
import factory
import tests
import performance


# Just to silent an harmless warning
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

mac_BF_env = factory.CreateEnvironment()

# tests.test_centralized_controller(mac_BF_env)

# tests.test_decentralized_controller(mac_BF_env)

# tests.test_sim_controller(mac_BF_env)

# tests.test_coordinator(mac_BF_env)

tests.test_sim_coordinator(mac_BF_env)


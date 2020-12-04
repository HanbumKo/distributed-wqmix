
use_wandb = True
use_tensorboard = False
device = 'cuda:1'

map_name = "3s5z"

epsilon_start = 1.0
epsilon_finish = 0.05
epsilon_anneal_time = 1000000
# epsilon_anneal_time = 50000
buffer_size = 5000
target_update_interval = 200 # update the target network every {} episodes
train_num = 1000000
test_interval = 30
mixing_embed_dim = 32
hypernet_layers = 2
hypernet_embed = 64

wqmix = True
w = 0.75
double_q = True
hysteretic_qmix = False

gamma = 0.99
batch_size = 32 # Number of episodes to train on
lr = 0.0005 # Learning rate for agents
optim_alpha = 0.99 # RMSProp alpha
optim_eps = 0.00001 # RMSProp epsilon
grad_norm_clip = 10 # Reduce magnitude of gradients above this L2 norm


rnn_hidden_dim = 64 # Size of hidden state for default rnn agent


# Ray
n_cpus = 20



from smac.env import StarCraft2Env
env = StarCraft2Env(map_name=map_name)
env_info = env.get_env_info()
n_actions = env_info["n_actions"]
n_agents = env_info["n_agents"]
state_shape = env_info["state_shape"]
obs_shape = env_info["obs_shape"]
input_shape = obs_shape + n_actions + n_agents
episode_limit = env_info["episode_limit"]

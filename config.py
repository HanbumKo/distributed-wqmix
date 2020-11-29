
map_name = "8m"

epsilon_start = 1.0
epsilon_finish = 0.05
# epsilon_anneal_time = 50000
epsilon_anneal_time = 20
buffer_size = 5000
target_update_interval = 200 # update the target network every {} episodes
mixing_embed_dim = 32
hypernet_layers = 2
hypernet_embed = 64

w = 0.5
double_q = True
hysteretic_qmix = True


gamma = 0.99
batch_size = 32 # Number of episodes to train on
buffer_size = 32 # Size of the replay buffer
lr = 0.0005 # Learning rate for agents
optim_alpha = 0.99 # RMSProp alpha
optim_eps = 0.00001 # RMSProp epsilon
grad_norm_clip = 10 # Reduce magnitude of gradients above this L2 norm


rnn_hidden_dim = 64 # Size of hidden state for default rnn agent
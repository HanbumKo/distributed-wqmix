import config

from worker import RolloutWorker
from wq_learner import WQLearner
from replay_buffer import ReplayBuffer
from smac.env import StarCraft2Env


env = StarCraft2Env(map_name=config.map_name)
env_info = env.get_env_info()
config.n_actions = env_info["n_actions"]
config.n_agents = env_info["n_agents"]
config.state_shape = env_info["state_shape"]
config.obs_shape = env_info["obs_shape"]
config.input_shape = config.obs_shape + config.n_actions
config.episode_limit = env_info["episode_limit"]


evaluate = False
worker = RolloutWorker()
buf = ReplayBuffer()
learner = WQLearner()

for _ in range(10000):
    # Synchronize
    worker.set_rnnagent_state_dict(learner.get_rnnagent_state_dict())

    episode, episode_reward, win_tag = worker.generate_episode(evaluate=evaluate)
    buf.store_epi(episode)

    batch = buf.sample_batch(batch_size=config.batch_size)
    learner.learn(batch)
    print()
    print("="*60)
    print("episode_reward :", episode_reward)
    print("win :", win_tag)
    print("="*60)
    print()

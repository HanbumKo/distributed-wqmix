import config
import ray
import numpy as np

from worker import RolloutWorker
from wq_learner import WQLearner
from q_learner import QLearner
from replay_buffer import ReplayBuffer
from smac.env import StarCraft2Env


# env = StarCraft2Env(map_name=config.map_name)
# env_info = env.get_env_info()
# config.n_actions = env_info["n_actions"]
# config.n_agents = env_info["n_agents"]
# config.state_shape = env_info["state_shape"]
# config.obs_shape = env_info["obs_shape"]
# config.input_shape = config.obs_shape + config.n_actions
# config.episode_limit = env_info["episode_limit"]



ray.init(num_cpus=config.n_cpus)

if config.use_wandb:
    import wandb
    wandb.init(project="wqmix_test", entity="sai", name="CWQMIX")


workers = [RolloutWorker.remote() for _ in range(config.n_cpus)]
buf = ReplayBuffer()
learner = WQLearner()

for i in range(config.train_num):
    # Synchronize
    for worker in workers:
        ray.get(worker.set_rnnagent_state_dict.remote(learner.get_rnnagent_state_dict()))

    worker_results = []
    # episode_ops = [worker.generate_episode.remote(evaluate=False) for worker in workers]
    # worker_outputs = ray.get(episode_ops)
    # for worker_output in worker_outputs:
    #     buf.store_epi(worker_output[0])
    #     worker_results.append(worker_output[1])

     # Store only one episode
    worker_output = ray.get(workers[0].generate_episode.remote(evaluate=False))
    buf.store_epi(worker_output[0])
    worker_results.append(worker_output[1])

    batch = buf.sample_batch(batch_size=config.batch_size)

    train_results = []
    # for _ in range(config.n_cpus):
    for _ in range(1):
        train_results.append(learner.learn(batch))

    episode_rewards = [worker_result["episode_reward"] for worker_result in worker_results]
    wins = [worker_result["win_tag"] for worker_result in worker_results]
    episode_lens = [worker_result["episode_len"] for worker_result in worker_results]
    epsilons = [worker_result["epsilon"] for worker_result in worker_results]
    losses = [train_result["loss"] for train_result in train_results]
    train_steps = [train_result["train_step"] for train_result in train_results]
    masked_td_errors = [train_result["masked_td_error"] for train_result in train_results]

    print()
    print("="*60)
    print("episode_reward :", np.mean(episode_rewards))
    print("win :", np.mean(wins))
    print("episode_len :", np.mean(episode_lens))
    print("epsilon :", np.mean(epsilons))
    print("loss :", np.mean(losses))
    print("train_step :", np.mean(train_steps))
    print("masked_td_error :", np.mean(masked_td_errors))
    print("buffer size :", buf.size)
    print("="*60)
    print()

    if config.use_wandb:
        wandb.log({"Train/episode_reward": np.mean(episode_rewards),
                   "Train/win": np.mean(wins),
                   "Train/episode_len": np.mean(episode_lens),
                   "Train/epsilon": np.mean(epsilons),
                   "loss": np.mean(losses),
                   "train_step": np.mean(train_steps),
                   "masked_td_error": np.mean(masked_td_errors),
                   "buffer size": buf.size})


    # Evaluate
    if i % config.test_interval == 0:
        worker_results = []
        episode_ops = [worker.generate_episode.remote(evaluate=True) for worker in workers]
        worker_outputs = ray.get(episode_ops)
        for worker_output in worker_outputs:
            worker_results.append(worker_output[1])

        episode_rewards = [worker_result["episode_reward"] for worker_result in worker_results]
        wins = [worker_result["win_tag"] for worker_result in worker_results]
        episode_lens = [worker_result["episode_len"] for worker_result in worker_results]

        if config.use_wandb:
            wandb.log({"Test/avg_episode_reward": np.mean(episode_rewards),
                       "Test/win_rate": np.mean(wins),
                       "Test/avg_episode_len": np.mean(episode_lens)})


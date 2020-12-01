import config
import torch
import ray
import numpy as np

from smac.env import StarCraft2Env
from model import RNNAgent, QMixer


@ray.remote
class RolloutWorker():
    def __init__(self):
        self.rnnagent = RNNAgent()
        self.epsilon = config.epsilon_start
        self.epsilon_deg = (config.epsilon_start - config.epsilon_finish) / config.epsilon_anneal_time

        self.env = StarCraft2Env(map_name=config.map_name)
    
    def get_epsilon(self):
        returned_eps = self.epsilon
        self.epsilon = self.epsilon - (self.epsilon_deg*config.n_cpus) if self.epsilon > config.epsilon_finish else self.epsilon
        self.epsilon = max(self.epsilon, config.epsilon_finish)
        return returned_eps

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate, hidden):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose
        inputs = np.hstack((inputs, last_action))
        hidden_state = hidden[:, agent_num, :]

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

        # get q value
        q_value, hidden[:, agent_num, :] = self.rnnagent(inputs, hidden_state)

        q_value[avail_actions == 0.0] = -float("inf")
        if np.random.uniform() < epsilon and not evaluate:
            action = np.random.choice(avail_actions_ind)
        else:
            action = torch.argmax(q_value)

        return action

    def generate_episode(self, evaluate):
        o_dim = config.obs_shape
        s_dim = config.state_shape
        size = config.buffer_size
        max_epi_len = config.episode_limit
        n_agents = config.n_agents
        n_actions = config.n_actions
        o        = np.zeros((max_epi_len, n_agents, o_dim), dtype=np.float32)
        u        = np.zeros((max_epi_len, n_agents, 1), dtype=np.float32)
        s        = np.zeros((max_epi_len, s_dim), dtype=np.float32)
        r        = np.zeros((max_epi_len, 1), dtype=np.float32)
        d        = np.ones((max_epi_len, 1), dtype=np.float32)
        pad      = np.ones((max_epi_len, 1), dtype=np.float32)
        o2       = np.zeros((max_epi_len, n_agents, o_dim), dtype=np.float32)
        s2       = np.zeros((max_epi_len, s_dim), dtype=np.float32)
        avail_u  = np.zeros((max_epi_len, n_agents, n_actions), dtype=np.float32)
        avail_u2 = np.zeros((max_epi_len, n_agents, n_actions), dtype=np.float32)
        u_onehot = np.zeros((max_epi_len, n_agents, n_actions), dtype=np.float32)

        self.env.reset()
        done = False
        win_tag = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((config.n_agents, config.n_actions))
        hidden = torch.zeros((1, config.n_agents, config.rnn_hidden_dim))

        epsilon = self.get_epsilon()

        while not done and step < max_epi_len:
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions = []
            for agent_id in range(config.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                            avail_action, epsilon, evaluate, hidden)
                action_onehot = np.zeros(config.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                u[step, agent_id, 0] = action
                u_onehot[step, agent_id, :] = action_onehot
                avail_u[step, agent_id, :] = avail_action
                last_action[agent_id] = action_onehot
            
            reward, done, info = self.env.step(actions)
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            o[step, :, :] = np.array(obs)
            s[step, :] = np.array(state)
            r[step, 0] = reward
            d[step, 0] = done
            pad[step, 0] = 0
            episode_reward += reward

            # next obs, state, action
            obs2 = self.env.get_obs()
            state2 = self.env.get_state()
            o2[step, :, :] = np.array(obs2)
            s2[step, :] = np.array(state)
            for agent_id in range(config.n_agents):
                avail_action2 = self.env.get_avail_agent_actions(agent_id)
                avail_u2[step, agent_id, :] = avail_action2
            
            step += 1
            
        episode = dict(o=o,
                       u=u,
                       s=s,
                       r=r,
                       d=d,
                       pad=pad,
                       o2=o2,
                       s2=s2,
                       avail_u=avail_u,
                       avail_u2=avail_u2,
                       u_onehot=u_onehot)
        
        win_tag = 1 if win_tag else 0
        results = dict(episode_reward=episode_reward,
                       win_tag=win_tag,
                       episode_len=step,
                       epsilon=epsilon)
        
        return episode, results

    def set_rnnagent_state_dict(self, rnnagent_state_dict):
        self.rnnagent.load_state_dict(rnnagent_state_dict)
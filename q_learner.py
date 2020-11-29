import config
import torch
import numpy as np

from model import RNNAgent, QMixer


class WQLearner():
    def __init__(self):
        self.rnnagent = RNNAgent()
        self.rnnagent_targ = RNNAgent()
        self.mixer = QMixer()
        self.mixer_targ = QMixer()

        self.rnnagent_targ.load_state_dict(self.rnnagent.state_dict())
        self.mixer_targ.load_state_dict(self.mixer.state_dict())

        self.params = list(self.rnnagent.parameters()) + list(self.mixer.parameters())
        self.optim = torch.optim.RMSprop(self.params, lr=config.lr)

        self.train_step = 0
    
    def get_rnnagent_state_dict(self):
        return self.rnnagent.state_dict()

    def learn(self, batch):
        max_episode_len = self.get_max_episode_len(batch)
        episode_num = batch['o'].shape[0]
        self.rnn_hidden = torch.zeros((episode_num, config.n_agents, config.rnn_hidden_dim))
        self.rnn_targ_hidden = torch.zeros((episode_num, config.n_agents, config.rnn_hidden_dim))
        batch['u'] = torch.tensor(batch['u'], dtype=torch.long)

        s, s2, u, r, avail_u, avail_u2, d = batch['s'], batch['s2'], batch['u'], \
                                            batch['r'],  batch['avail_u'], batch['avail_u2'],\
                                            batch['d']
        s, s2, u, r, avail_u, avail_u2, d = s[:, :max_episode_len, :], s2[:, :max_episode_len, :],\
                                            u[:, :max_episode_len, :, :], r[:, :max_episode_len, :],\
                                            avail_u[:, :max_episode_len, :, :], avail_u2[:, :max_episode_len, :, :],\
                                            d[:, :max_episode_len, :]
        mask = 1 - batch["pad"][:, :max_episode_len, :].float()

        q_values, q_targ_values = self.get_q_values(batch, max_episode_len)
        q_values = torch.gather(q_values, dim=3, index=u).squeeze(3)

        q_targ_values[avail_u2 == 0.0] = -9999999
        q_targ_values = q_targ_values.max(dim=3)[0]

        q_total = self.mixer(q_values, s)
        q_total_targ = self.mixer_targ(q_targ_values, s2)

        targets = r + config.gamma * q_total_targ * (1 - d)
        
        td_error = (q_total - targets.detach())
        masked_td_error = mask * td_error

        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, config.grad_norm_clip)
        self.optim.step()

        if self.train_step > 0 and self.train_step % config.target_update_interval == 0:
            self.rnnagent_targ.load_state_dict(self.rnnagent.state_dict())
            self.mixer_targ.load_state_dict(self.mixer.state_dict())
        
        self.train_step += 1
    
    def get_q_values(self, batch, max_episode_len):
        batch_size = batch['o'].shape[0]
        q_values, q_targ_values = [], []
        for time_step in range(max_episode_len):
            inputs, inputs_next = self.get_inputs(batch, time_step)
            q_value, self.rnn_hidden = self.rnnagent(inputs, self.rnn_hidden)
            q_targ_value, self.rnn_targ_hidden = self.rnnagent_targ(inputs_next, self.rnn_targ_hidden)

            q_value = q_value.view(batch_size, config.n_agents, -1)
            q_targ_value = q_targ_value.view(batch_size, config.n_agents, -1)
            q_values.append(q_value)
            q_targ_values.append(q_targ_value)
        q_values = torch.stack(q_values, dim=1)
        q_targ_values = torch.stack(q_targ_values, dim=1)
        
        return q_values, q_targ_values

    def get_inputs(self, batch, time_step):
        o, o2, u_onehot = batch['o'][:, time_step], batch['o2'][:, time_step], batch['u_onehot'][:]
        batch_size = o.shape[0]
        inputs, inputs_next = [], []
        inputs.append(o)
        inputs_next.append(o2)
        if time_step == 0:
            inputs.append(torch.zeros_like(u_onehot[:, time_step]))
        else:
            inputs.append(u_onehot[:, time_step - 1])
        inputs_next.append(u_onehot[:, time_step])
        inputs = torch.cat([x.reshape(batch_size * config.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(batch_size * config.n_agents, -1) for x in inputs_next], dim=1)
        
        return inputs, inputs_next

    def get_max_episode_len(self, batch):
        d = batch['d']
        batch_size = d.shape[0]
        max_episode_len = 0
        for batch_idx in range(batch_size):
            for time_step in range(config.episode_limit):
                if d[batch_idx, time_step, 0] == 1:
                    if time_step + 1 >= max_episode_len:
                        max_episode_len = time_step + 1
                    break

        return max_episode_len

        
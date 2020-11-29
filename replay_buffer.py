import torch
import config
import numpy as np

class ReplayBuffer:
    def __init__(self):
        o_dim = config.obs_shape
        s_dim = config.state_shape
        size = config.buffer_size
        max_epi_len = config.episode_limit
        n_agents = config.n_agents
        n_actions = config.n_actions
        self.o_buf        = np.zeros((size, max_epi_len, n_agents, o_dim), dtype=np.float32)
        self.u_buf        = np.zeros((size, max_epi_len, n_agents, 1), dtype=np.float32)
        self.s_buf        = np.zeros((size, max_epi_len, s_dim), dtype=np.float32)
        self.r_buf        = np.zeros((size, max_epi_len, 1), dtype=np.float32)
        self.d_buf        = np.zeros((size, max_epi_len, 1), dtype=np.float32)
        self.pad_buf      = np.zeros((size, max_epi_len, 1), dtype=np.float32)
        self.o2_buf       = np.zeros((size, max_epi_len, n_agents, o_dim), dtype=np.float32)
        self.s2_buf       = np.zeros((size, max_epi_len, s_dim), dtype=np.float32)
        self.avail_u_buf  = np.zeros((size, max_epi_len, n_agents, n_actions), dtype=np.float32)
        self.avail_u2_buf = np.zeros((size, max_epi_len, n_agents, n_actions), dtype=np.float32)
        self.u_onehot_buf = np.zeros((size, max_epi_len, n_agents, n_actions), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    # def store_epi(self, o, u, s, r, d, pad, o2, s2, avail_u, avail_u2, u_onehot):
    def store_epi(self, episode):
        self.o_buf[self.ptr, :, :, :] = episode['o']
        self.u_buf[self.ptr, :, :, :] = episode['u']
        self.s_buf[self.ptr, :, :] = episode['s']
        self.r_buf[self.ptr, :, :] = episode['r']
        self.d_buf[self.ptr, :, :] = episode['d']
        self.pad_buf[self.ptr, :, :] = episode['pad']
        self.o2_buf[self.ptr, :, :, :] = episode['o2']
        self.s2_buf[self.ptr, :, :] = episode['s2']
        self.avail_u_buf[self.ptr, :, :, :] = episode['avail_u']
        self.avail_u2_buf[self.ptr, :, :, :] = episode['avail_u2']
        self.u_onehot_buf[self.ptr, :, :, :] = episode['u_onehot']
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(o=self.o_buf[idxs],
                     u=self.u_buf[idxs],
                     s=self.s_buf[idxs],
                     r=self.r_buf[idxs],
                     d=self.d_buf[idxs],
                     pad=self.pad_buf[idxs],
                     o2=self.o2_buf[idxs],
                     s2=self.s2_buf[idxs],
                     avail_u=self.avail_u_buf[idxs],
                     avail_u2=self.avail_u2_buf[idxs],
                     u_onehot=self.u_onehot_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
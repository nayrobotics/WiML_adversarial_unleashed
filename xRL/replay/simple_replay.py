import torch

# Implementation of ring buffer for trajectories
# ReplayBuffer is used to store trajectories for off-policy learning
# SimplifiedReplayBuffer is used for simplified storage of observations (AMP) 

def create_buffer(capacity, obs_dim, action_dim, device='cuda'):
    if isinstance(capacity, int):
        capacity = (capacity,)

    buf_obs_size = (*capacity, obs_dim) if isinstance(obs_dim, int) else (*capacity, *obs_dim)
    buf_obs = torch.zeros(buf_obs_size,
                          dtype=torch.float32, device=device)
    buf_action = torch.zeros((*capacity, int(action_dim)),
                             dtype=torch.float32, device=device)
    buf_reward = torch.zeros((*capacity, 1),
                             dtype=torch.float32, device=device)
    buf_next_obs = torch.zeros(buf_obs_size,
                               dtype=torch.float32, device=device)
    buf_done = torch.zeros((*capacity, 1),
                           dtype=torch.bool, device=device)
    return buf_obs, buf_action, buf_next_obs, buf_reward, buf_done


def create_simplified_buffer(capacity, obs_dim, device='cuda'):
    if isinstance(capacity, int):
        capacity = (capacity,)
    buf_obs_size = (*capacity, obs_dim) if isinstance(obs_dim, int) else (*capacity, *obs_dim)
    buf_obs = torch.zeros(buf_obs_size,
                          dtype=torch.float32, device=device)
    buf_next_obs = torch.zeros(buf_obs_size,
                               dtype=torch.float32, device=device)

    return buf_obs, buf_next_obs



class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device='cpu'):
        self.obs_dim = obs_dim
        if isinstance(obs_dim, int):
            self.obs_dim = (self.obs_dim,)
        self.action_dim = action_dim
        self.device = device
        self.step = 0  # next pointer
        self.if_full = False
        
        self.cur_capacity = 0  # current capacity
        self.capacity = int(capacity)


        ret = create_buffer(capacity=self.capacity, obs_dim=obs_dim, action_dim=action_dim, device=device)
        self.buf_obs, self.buf_action, self.buf_next_obs, self.buf_reward, self.buf_done = ret

    @torch.no_grad()
    def add_to_buffer(self, trajectory):
        obs, actions, rewards, next_obs, dones = trajectory
        
        obs = obs.reshape(-1, *self.obs_dim).to(self.device)
        actions = actions.reshape(-1, self.action_dim).to(self.device)
        rewards = rewards.reshape(-1, 1).to(self.device)
        next_obs = next_obs.reshape(-1, *self.obs_dim).to(self.device)
        dones = dones.reshape(-1, 1).bool().to(self.device)
       
        num_states = rewards.shape[0]
        start_idx = self.step
        end_idx = start_idx + num_states

        if end_idx > self.capacity:
            self.if_full = True

            remaining = self.capacity - self.step

            self.buf_obs[self.step:self.capacity] = obs[:remaining]
            self.buf_action[self.step:self.capacity] = actions[:remaining]
            self.buf_reward[self.step:self.capacity] = rewards[:remaining]
            self.buf_next_obs[self.step:self.capacity] = next_obs[:remaining]
            self.buf_done[self.step:self.capacity] = dones[:remaining]

            wrap_around = end_idx - self.capacity

            self.buf_obs[:wrap_around] = obs[remaining:]
            self.buf_action[:wrap_around] = actions[remaining:]
            self.buf_reward[:wrap_around] = rewards[remaining:]
            self.buf_next_obs[:wrap_around] = next_obs[remaining:]
            self.buf_done[:wrap_around] = dones[remaining:]
        else:
            self.buf_obs[self.step:end_idx] = obs
            self.buf_action[self.step:end_idx] = actions
            self.buf_reward[self.step:end_idx] = rewards
            self.buf_next_obs[self.step:end_idx] = next_obs
            self.buf_done[self.step:end_idx] = dones

        self.step = (self.step + num_states) % self.capacity
        self.cur_capacity = min(self.capacity, max(end_idx, self.cur_capacity))

        print(f"\n Replay Buffer Current capacity: {self.cur_capacity}")

    @torch.no_grad()
    def sample_batch(self, batch_size, device='cuda'):
        indices = torch.randint(self.cur_capacity, size=(batch_size,), device=self.device)
        return (
            self.buf_obs[indices].to(device),
            self.buf_action[indices].to(device),
            self.buf_reward[indices].to(device),
            self.buf_next_obs[indices].to(device),
            self.buf_done[indices].to(device).float()
        )


class SymplifiedReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, num_envs:int, device='cpu'):
        self.obs_dim = obs_dim
        if isinstance(obs_dim, int):
            self.obs_dim = (self.obs_dim,)
        self.device = device
        self.step = 0  # next pointer
        self.if_full = False
        self.cur_capacity = 0  # current capacity
        self.capacity = int(capacity)
        self.num_envs = num_envs
        self.step = 0
        self.num_samples = 0

        ret = create_simplified_buffer(capacity=self.capacity, obs_dim=obs_dim, device=device)
        self.buf_obs, self.buf_next_obs = ret

    @torch.no_grad()
    def add_to_buffer(self, trajectory):

        obs, next_obs = trajectory
        obs = obs.reshape(-1, *self.obs_dim).to(self.device)

        next_obs = next_obs.reshape(-1, *self.obs_dim).to(self.device)
        
        num_states = obs.shape[0]
        start_idx = self.step
        end_idx = start_idx + num_states
        
        if end_idx > self.capacity:
            self.if_full = True
            
            remaining = self.capacity - self.step

            self.buf_obs[self.step:self.capacity] = obs[:remaining]
            self.buf_next_obs[self.step:self.capacity] = next_obs[:remaining]
            
            wrap_around = end_idx - self.capacity
            
            self.buf_obs[:wrap_around] = obs[remaining:]
            self.buf_next_obs[:wrap_around] = next_obs[remaining:]
            
        else:
            self.buf_obs[self.step:end_idx] = obs
            self.buf_next_obs[self.step:end_idx] = next_obs

        #self.step = end_idx  # update pointer
        self.step = (self.step + num_states) % self.capacity

        #self.cur_capacity = self.capacity if self.if_full else self.step
        self.cur_capacity = min(self.capacity, max(end_idx, self.cur_capacity))

        #self.save_debug_csv("debug_buffer_snapshot.csv")

        print(f"\n Simplified Replay Buffer Current capacity: {self.cur_capacity}")


    @torch.no_grad()
    def sample_batch(self, batch_size, device='cuda'):
        indices = torch.randint(self.cur_capacity, size=(batch_size,), device=self.device)
        return (
            self.buf_obs[indices].to(device),
            self.buf_next_obs[indices].to(device),
        )

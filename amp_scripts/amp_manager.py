import os
import glob
import torch

from typing import Any
from omegaconf.dictconfig import DictConfig

from dataclasses import dataclass

from .amp_discriminator import AMPDiscriminator

from .motion_loader import AMPLoader
from xRL.utils.torch_util import Normalizer

from xRL import ROOT_DIR

@dataclass
class AMP:
    env: Any
    cfg: DictConfig

    def __post_init__(self):
        #Manage amp load and discriminator

        self.device = torch.device(self.cfg.algo.amp_device)

        min_normalized_std = list(self.cfg.algo.min_normalized_std) * 4
        min_std = (
            torch.tensor(min_normalized_std, device=self.device) *
            (torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0])))        

        motion_path = self.env.amp_files_path
        
        #Save motions frames on memory preload_transitions=True 
        amp_data = AMPLoader(
            self.device,  time_between_frames=self.env.dt, preload_transitions=True,
            num_preload_transitions=self.cfg.algo.amp_num_preload_transitions,
            motion_files=motion_path)
        
        amp_normalizer = Normalizer(amp_data.observation_dim)
        
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            self.cfg.algo.amp_reward_coef,
            self.cfg.algo.amp_discr_hidden_dims, self.device,
            self.cfg.algo.amp_task_reward_lerp).to(self.device)


        #Amp components
        self.min_std = min_std

        # Discriminator components
        self.discriminator = discriminator
        self.discriminator.to(self.device)

        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

    
    def compute_amp_loss(self, policy_state, policy_next_state, expert_state, expert_next_state):
        #Copyright (c) 2021, ETH Zurich, Nikita Rudin
        #Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES
        #https://github.com/escontra/AMP_for_hardware/tree/main
        
        policy_state_unnorm = torch.clone(policy_state)
        expert_state_unnorm = torch.clone(expert_state)

        if self.amp_normalizer is not None:
            with torch.no_grad():
                        policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                        policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                        expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                        expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
            expert_loss = torch.nn.MSELoss()(
                    expert_d, torch.ones(expert_d.size(), device=self.device))
            policy_loss = torch.nn.MSELoss()(
                    policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.discriminator.compute_grad_pen(
                    expert_state, expert_next_state, lambda_=10)


        if self.amp_normalizer is not None:
            self.amp_normalizer.update(policy_state_unnorm)
            self.amp_normalizer.update(expert_state_unnorm)
        

        return amp_loss.item(), grad_pen_loss.item(), policy_d, expert_d

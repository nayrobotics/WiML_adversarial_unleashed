#import gc

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from xRL.algo.ac_base import ActorCriticBase
from xRL.replay.nstep_replay import NStepReplay

from xRL.utils.common import handle_timeout
from xRL.utils.torch_util import soft_update, memory_manegement

from amp_scripts.amp_manager import AMP


@dataclass
class AgentAMPSAC(ActorCriticBase):

    def __post_init__(self):
        super().__post_init__() #Initialize SAC and Actor Critic
        print("AgentAMPSAC")

        self.critic_target = deepcopy(self.critic)
        self.actor_target = deepcopy(self.actor) if not self.cfg.algo.no_tgt_actor else self.actor

        self.amp_obs_dim = self.env.amp_observation_dim
        self.amp_loss_coef = self.cfg.algo.amp_loss_coef        
        self.grad_pen_loss_coef = self.cfg.algo.grad_pen_loss_coef

        self.obs = None
        if self.cfg.algo.alpha is None:
            self.log_alpha = nn.Parameter(torch.zeros(1, device=self.device))
            self.alpha_optim = torch.optim.AdamW([self.log_alpha], lr=self.cfg.algo.alpha_lr)

        self.target_entropy = -self.action_dim

        self.n_step_buffer = NStepReplay(obs_dim=self.obs_dim,
                                         action_dim=self.action_dim,
                                         amp_dim=self.amp_obs_dim,
                                         num_envs=self.cfg.num_envs,
                                         nstep=self.cfg.algo.nstep,
                                         device=self.device,
                                         gamma=self.cfg.algo.gamma)
        

        #Initialize AMP manager
        self.amp_manager = AMP(env = self.env, cfg = self.cfg)

        self.amp_device = self.amp_manager.device
        
        params = [
                {'params': self.actor.parameters(), 'name': 'actor'},
                {'params': self.amp_manager.discriminator.trunk.parameters(),
                'weight_decay': 10e-4, 'name': 'amp_trunk'},
                {'params': self.amp_manager.discriminator.amp_linear.parameters(),
                'weight_decay': 10e-2, 'name': 'amp_head'}]
        
        #Replace actor optimizer considering AMP parameters                       
        self.actor_optimizer = torch.optim.AdamW(params, self.cfg.algo.actor_lr)

    def get_alpha(self, detach=True, scalar=False):
        if self.cfg.algo.alpha is None:
            alpha = self.log_alpha.exp()
            if detach:
                alpha = alpha.detach()
            if scalar:
                alpha = alpha.item()
        else:
            alpha = self.cfg.algo.alpha
        return alpha

    
    def get_actions(self, obs, sample=True):
        if self.cfg.algo.obs_norm:
            obs = self.obs_rms.normalize(obs)
        actions = self.actor.get_actions(obs, sample=sample)
        return actions 

    @torch.no_grad()
    def explore_env(self, env, timesteps: int, random: bool) -> list:
        
        obs_dim = (self.obs_dim,) if isinstance(self.obs_dim, int) else self.obs_dim

        traj_obs = torch.empty((self.cfg.num_envs, timesteps) + (*obs_dim,), device=self.device)
        traj_actions = torch.empty((self.cfg.num_envs, timesteps) + (self.action_dim,), device=self.device)
        traj_rewards = torch.empty((self.cfg.num_envs, timesteps), device=self.device)
        traj_next_obs = torch.empty((self.cfg.num_envs, timesteps) + (*obs_dim,), device=self.device)
        traj_dones = torch.empty((self.cfg.num_envs, timesteps), device=self.device)

        amp_obs_dim = (self.amp_obs_dim,) if isinstance(self.amp_obs_dim, int) else self.amp_obs_dim

        amp_traj_obs = torch.empty((self.cfg.num_envs, timesteps) + (*amp_obs_dim,), device=self.amp_device)
        amp_traj_next_obs = torch.empty((self.cfg.num_envs, timesteps) + (*amp_obs_dim,), device=self.amp_device)


        ep_infos = []    
        obs = self.obs
        amp_obs = self.env.get_amp_observations()


        for i in range(timesteps):
            if self.cfg.algo.obs_norm:
                self.obs_rms.update(obs)
            if random:
                action = torch.rand((self.cfg.num_envs, self.action_dim),
                                    device=self.device) * 2.0 - 1.0
            else:
                action = self.get_actions(obs, sample=True)

            next_obs, reward, done, info, reset_env_ids, terminal_amp_states = env.step(action)

            # Get AMP obs
            next_amp_obs = self.env.get_amp_observations()

            # Account for terminal states.
            next_amp_obs_with_term = torch.clone(next_amp_obs)
            next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

            reward = self.amp_manager.discriminator.predict_amp_reward(
            amp_obs, next_amp_obs_with_term, reward, normalizer=self.amp_manager.amp_normalizer)[0]

            self.update_tracker(reward, done, info)

            disc_reward = self.amp_manager.discriminator.reward_discrim
            self.update_additional_tracker(disc_reward, done)

            if self.cfg.algo.handle_timeout:
                done = handle_timeout(done, info)

            if 'episode' in info:
                ep_infos.append(info['episode'])            

            traj_obs[:, i] = obs.to(self.device)
            traj_actions[:, i] = action.to(self.device)
            traj_dones[:, i] = done.to(self.device)
            traj_rewards[:, i] = reward.to(self.device)
            traj_next_obs[:, i] = next_obs.to(self.device)
            
            amp_traj_obs[:, i] = amp_obs.to(self.amp_device)            
            amp_traj_next_obs[:, i] = next_amp_obs.to(self.amp_device)

            obs = next_obs
            amp_obs = torch.clone(next_amp_obs)

        self.obs = obs

        traj_rewards = self.cfg.algo.reward_scale * traj_rewards.reshape(self.cfg.num_envs, timesteps, 1)
        traj_dones = traj_dones.reshape(self.cfg.num_envs, timesteps, 1).to(self.device)
        data = self.n_step_buffer.add_to_buffer(traj_obs, traj_actions, traj_rewards, traj_next_obs, traj_dones, amp_traj_obs, amp_traj_next_obs)
        
        # release memory variables
        del traj_obs, traj_actions, traj_rewards, traj_dones, traj_next_obs
        del amp_traj_obs, amp_traj_next_obs
        #gc.collect()
        torch.cuda.empty_cache()

        return data, timesteps * self.cfg.num_envs, ep_infos

    def update_net(self, memory):
        memory_manegement()

        critic_loss_list = list()
        actor_loss_list = list()
        # AMP
        amp_loss_list = list()
        grad_pen_loss_list = list()
        # Discriminator predictions
        policy_d_list = list()

        num_updates = self.cfg.algo.update_times

        for i in range(num_updates):
                
            obs, action, reward, next_obs, done, amp_obs, amp_next_obs = memory.sample_batch(self.cfg.algo.batch_size)
            obs, action, reward, next_obs, done, amp_obs, amp_next_obs = obs.to(self.device), action.to(self.device), reward.to(self.device), next_obs.to(self.device), done.to(self.device), amp_obs.to(self.amp_device), amp_next_obs.to(self.amp_device)

            policy_state, policy_next_state = amp_obs.clone(), amp_next_obs.clone()

            #from amp_data
            expert_state, expert_next_state = self.amp_manager.amp_data.feed_forward_generator(self.cfg.algo.amp_batch_count, self.cfg.algo.amp_batch_size)
            
            # Compute AMP loss and update AMP discriminator 
            amp_loss, grad_pen_loss, policy_d, expert_d = self.amp_manager.compute_amp_loss(policy_state, policy_next_state, expert_state, expert_next_state)
            amp_loss_list.append(amp_loss)
            grad_pen_loss_list.append(grad_pen_loss)
            policy_d_list.append(policy_d)

            if self.cfg.algo.obs_norm:
                obs = self.obs_rms.normalize(obs)
                next_obs = self.obs_rms.normalize(next_obs)
            critic_loss = self.update_critic(obs, action, reward, next_obs, done)
            critic_loss_list.append(critic_loss)

            actor_loss = self.update_actor(obs, amp_loss, grad_pen_loss)
            actor_loss_list.append(actor_loss)

            soft_update(self.critic_target, self.critic, self.cfg.algo.tau)
            if not self.cfg.algo.no_tgt_actor:
                soft_update(self.actor_target, self.actor, self.cfg.algo.tau)

        policy_d_tensor = torch.cat([d.detach().cpu() for d in policy_d_list])
        log_info = {
            "train/critic_loss": np.mean(critic_loss_list),
            "train/actor_loss": np.mean(actor_loss_list),
            "train/return": self.return_tracker.mean(),
            "train/return_disc": self.disc_return_tracker.mean(),
            "train/episode_length": self.step_tracker.mean(),
            'train/alpha': self.get_alpha(scalar=True),
            'train/amp_loss' : np.mean(amp_loss_list),
            'train/grad_pen_loss' : np.mean(grad_pen_loss_list),
            'train/policy_d_mean': policy_d_tensor.mean().item(),
            'train/policy_d_min': policy_d_tensor.min().item(),
            'train/policy_d_max': policy_d_tensor.max().item()
        }
        self.add_info_tracker_log(log_info)

        #Release memory
        del obs, action, reward, next_obs, done, critic_loss, actor_loss
        del critic_loss_list, actor_loss_list, amp_loss_list, grad_pen_loss_list, policy_d_list
        del policy_state, policy_next_state, expert_state, expert_next_state, policy_d_tensor

        #gc.collect()

        torch.cuda.empty_cache()
        memory_manegement(allocator_config="")

        return log_info


    #SAC algo
    def update_actor(self, obs, amp_loss, grad_pen_loss):
        self.critic.requires_grad_(False)
        
        with torch.autocast(device_type='cuda'):  
            actions, _, log_prob = self.actor.get_actions_logprob(obs)
            Q = self.critic.get_q_min(obs, actions)
            actor_loss = (self.get_alpha() * log_prob - Q).mean()
            
            #Insert Discriminator loss 
            actor_loss = (actor_loss +  self.amp_loss_coef * amp_loss +  self.grad_pen_loss_coef * grad_pen_loss)

        self.optimizer_update_mix_precision(self.actor_optimizer, actor_loss)
        self.critic.requires_grad_(True)

        if self.cfg.algo.alpha is None:
            with torch.autocast(device_type='cuda'):  
                alpha_loss = (self.get_alpha(detach=False) * (-log_prob - self.target_entropy).detach()).mean()
            
            self.optimizer_update_mix_precision(self.alpha_optim, alpha_loss)

        return actor_loss.item()
    
    def update_critic(self, obs, action, reward, next_obs, done):
        with torch.no_grad():
            with torch.autocast(device_type='cuda'):  
                next_actions, _, log_prob = self.actor.get_actions_logprob(next_obs)
                target_Q = self.critic_target.get_q_min(next_obs, next_actions) - self.get_alpha() * log_prob
                target_Q = reward + (1 - done) * (self.cfg.algo.gamma ** self.cfg.algo.nstep) * target_Q

        with torch.autocast(device_type='cuda'):  
            current_Q1, current_Q2 = self.critic.get_q1_q2(obs, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Scale loss and perform backward pass
        self.optimizer_update_mix_precision(self.critic_optimizer, critic_loss)

        return critic_loss.item()
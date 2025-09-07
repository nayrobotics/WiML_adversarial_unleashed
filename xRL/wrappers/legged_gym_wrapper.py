from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, Union
import numpy as np
import os

@dataclass
class BaseEnvWrapper:
    """
    Generic wrapper for Isaac Gym environments that provides a standardized interface.
    """
    env: Any
    env_cfg: Dict[str, Any]
    log_path : str
    
    def __post_init__(self):
        self.observation_space = (self.env.num_obs,)
        self.action_space = (self.env.num_actions,)
        self.max_episode_length = int(self.env.max_episode_length)
        self.num_envs = self.env.num_envs
        self.dt = getattr(self.env, 'dt', 0.0)
        self.sim_time_step = getattr(self.env, 'sim_time_step', 0.0)
        self.dof_pos_limits = getattr(self.env, 'dof_pos_limits', None)
        self.sim_device = self.env.sim_device
        self.gym = self.env.gym
        self.sim = self.env.sim
        self.viewer = getattr(self.env, 'viewer', None)
        self.set_camera = self.env.set_camera
        self._reset_vars()
        self._update_status_vars()
        
    def _reset_vars(self):
        self.base_lin_vel_x = 0.0
        self.base_lin_vel_y = 0.0
        self.base_lin_vel_z = 0.0
        self.base_height = 0.0
        
    def _update_status_vars(self):
        if hasattr(self.env, 'base_lin_vel'):
            vel = self.env.base_lin_vel
            self.base_lin_vel_x = vel[:, :1].mean().item()
            self.base_lin_vel_y = vel[:, 1:2].mean().item()
            self.base_lin_vel_z = vel[:, 2:3].mean().item()
        if hasattr(self.env, 'base_height'):
            self.base_height = self.env.base_height.mean().item()
        if hasattr(self.env, 'root_states'):
            self.root_states = self.env.root_states
        if hasattr(self.env, 'dof_state'):
            self.dof_state = self.env.dof_state
        if hasattr(self.env, 'dof_pos'):
            self.dof_pos = self.env.dof_pos
        if hasattr(self.env, 'dof_vel'):
            self.dof_vel = self.env.dof_vel
    
    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        self._update_status_vars()
        return obs
    
    def get_obs(self) -> np.ndarray:
        return self.env.get_observations()
        
    def step(self, actions: np.ndarray, index : Union[int, None] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        next_obs, _, rewards, dones, env_info, *_ = self.env.step(actions)
        self._update_status_vars()
        if type(index) != int:
            index = slice(index)
        return next_obs, rewards, dones, env_info
    



@dataclass
class AMPEnvWrapper(BaseEnvWrapper):
    
    def __post_init__(self):
        super().__post_init__()        
        
        if hasattr(self.env, 'get_amp_observations'):
            self.amp_observation_dim = (self.get_amp_observations().shape[1],)
        else:
            self.amp_observation_dim = (0,)
        
        if hasattr(self.env, 'amp_config'):
            self.amp_config = self.env.amp_config
            self.amp_loader = self.env.amp_config.amp_loader
        
        self.amp_files_path = getattr(self.env, 'amp_motion_files_path', None)


    def _update_status_vars(self):
        super()._update_status_vars()
    
    def get_amp_observations(self) -> np.ndarray:
        return self.env.get_amp_observations()
    
    def step(self, actions: np.ndarray, index : Union[int, None] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, Optional[np.ndarray], Optional[Any]]:
        next_obs, _, rewards, dones, env_info, reset_env_ids, terminal_amp_states = self.env.step(actions)
        self._update_status_vars()
        if type(index) != int:
            index = slice(index)
        return next_obs, rewards, dones, env_info, reset_env_ids, terminal_amp_states

@dataclass
class LogEnvWrapper(BaseEnvWrapper):
    
    def __post_init__(self):
        super().__post_init__() 

        from ..utils.log_data import Entry

        self.logger = Entry(self.env.dt, self.log_path, self.env_cfg.init_state)

        self.set_camera = self.env.set_camera
        self._reset_vars()
        self._update_status_vars()
        
    def _reset_vars(self):
        self.base_lin_vel_x = 0.0
        self.base_lin_vel_y = 0.0
        self.base_lin_vel_z = 0.0
        self.base_height = 0.0

        
    def _update_status_vars(self):
        super()._update_status_vars()

    def reset(self) -> np.ndarray:
        super().reset()
    
    def _update_logger(self, actions, robot_index=0):
        self.logger.log_states(
                    {
                        'dt': self.env.dt,
                        'step': self.env.common_step_counter,
                        'step_time': self.env.common_step_counter * self.env.dt,

                        'dof_pos_target': (actions[robot_index] * self.env.cfg.control.action_scale).tolist(),
                        'dof_pos': self.env.dof_pos[robot_index].tolist(),
                        'dof_vel': self.env.dof_vel[robot_index].tolist(),
                        'dof_torque': self.env.torques[robot_index].tolist(),

                        'command_x': self.env.commands[robot_index, 0].mean().item(),
                        'command_y': self.env.commands[robot_index, 1].mean().item(),
                        'command_yaw': self.env.commands[robot_index, 2].mean().item(),

                        'base_vel_x': self.env.base_lin_vel[robot_index, 0].mean().item(),
                        'base_vel_y': self.env.base_lin_vel[robot_index, 1].mean().item(),
                        'base_vel_z': self.env.base_lin_vel[robot_index, 2].mean().item(),

                        'base_vel_roll': self.env.base_ang_vel[robot_index, 0].mean().item(),
                        'base_vel_pitch': self.env.base_ang_vel[robot_index, 1].mean().item(),
                        'base_vel_yaw': self.env.base_ang_vel[robot_index, 2].mean().item(),
                        'base_ort_x': self.env.projected_gravity[robot_index, 0].mean().item() ,
                        'base_ort_y': self.env.projected_gravity[robot_index, 1].mean().item() ,
                        'base_ort_z': self.env.projected_gravity[robot_index, 2].mean().item() ,

                        'contact_forces_z_0': self.env.contact_forces[robot_index, self.env.feet_indices[0], 2].cpu().numpy().item(),
                        'contact_forces_z_1': self.env.contact_forces[robot_index, self.env.feet_indices[1], 2].cpu().numpy().item(),
                        'contact_forces_z_2': self.env.contact_forces[robot_index, self.env.feet_indices[2], 2].cpu().numpy().item(),
                        'contact_forces_z_3': self.env.contact_forces[robot_index, self.env.feet_indices[3], 2].cpu().numpy().item(),


                        'foot_pos_x': self.env.foot_pos[robot_index, 0].cpu().numpy(),
                        'foot_pos_y': self.env.foot_pos[robot_index, 1].cpu().numpy(),
                        'foot_pos_z': self.env.foot_pos[robot_index, 2].cpu().numpy(),
                        'base_z': self.env.root_states[robot_index, 2].mean().item(),
                        'base_height': self.env.base_height[robot_index].mean().item(),
                    }
        )
        return self.logger.state_log    
    
    def step(self, actions, index):
        next_obs, _, rewards, dones, env_info, *_ = self.env.step(actions)
        self._update_status_vars()
        if type(index) != int:
            index = slice(index)
        self.logs = self._update_logger(actions, index)
        return next_obs, rewards, dones, env_info
    
    def save_log(self, filepath):
        self.logger._write_csv(filepath, self.logger.state_log)
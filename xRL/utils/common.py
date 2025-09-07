import ast
import platform
import random
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from datetime import timedelta

import os
import yaml
import gym
import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf, open_dict


def init_neptune(cfg):
    import neptune

    nepturn_config = cfg.logging.neptune
    run = neptune.init_run(**nepturn_config)
    
    return run

def manage_tracking(tracking, params=None):

    if tracking.neptune:
        run = init_neptune(params)
        run["parameters"] = params

    elif tracking.wandb:
        run = init_wandb(params)

    def update(tracking, stat=dict(), model=None, reward_config=None, policy_config=None):

        if tracking.neptune:

            if model != None:
                run["model"].upload(model)
            
            if reward_config != None:
                run["reward_config"].upload(reward_config)

            if policy_config != None:
                run["policy_config"].upload(policy_config)

            if "ct_seconds" in stat:
                run["ct_seconds"].append(stat["ct_seconds"])

            if "buffer_is_full_it" in stat:
                run["buffer_is_full_it"] = stat["buffer_is_full_it"]

            if "amp_buffer_is_full_it" in stat:
                run["amp_buffer_is_full_it"] = stat["amp_buffer_is_full_it"]

            if "global_steps" in stat:
                run["global_steps"].append(stat['global_steps'])

            if "episodes" in stat:
                run["episodes"].append(stat['episodes'])

            if "train/return" in stat:
                actor_loss = stat['train/actor_loss']
                critic_loss = stat['train/critic_loss']
                reward = stat['train/return']
                episode_lenght = stat['train/episode_length']
                
                run["rewards"].append(reward)
                run["episode_length"].append(episode_lenght)

                #loss
                if "train/alpha" in stat:
                    alpha = stat['train/alpha']
                    run["loss/alpha"].log(alpha)
                run["loss/actor"].append(actor_loss)
                run["loss/critic"].append(critic_loss)
            
            if "train/return_disc" in stat:
                run["reward_disc"].append(stat["train/return_disc"])

            if "eval/return" in stat:
                eval_reward = stat["eval/reward"]
                eval_episode_lenght = stat["eval/episode_length"]

                run["eval/episode_length"].append(eval_episode_lenght)
                run["eval/rewards"].append(eval_reward)

            
            if "train/amp_loss" in stat:
                amp_loss = stat["train/amp_loss"]
                grad_pen_loss = stat["train/grad_pen_loss"]
                run['loss/discriminator'].append(amp_loss)
                run['loss/grad_pen'].append(grad_pen_loss)

            if "train/policy_d_mean" in stat:
                run["amp_score/mean"].append(stat["train/policy_d_mean"])

            if "train/policy_d_min" in stat:
                run["amp_score/min"].append(stat["train/policy_d_min"])

            if "train/policy_d_max" in stat:
                run["amp_score/max"].append(stat["train/policy_d_max"])
            
            if "train/avg_lin_vel_x" in stat:
                run["metrics/avg_lin_vel_x"].append(stat["train/avg_lin_vel_x"])

            if "train/avg_lin_vel_y" in stat:
                run["metrics/avg_lin_vel_y"].append(stat["train/avg_lin_vel_y"])

            if "train/avg_lin_vel_z" in stat:
                run["metrics/avg_lin_vel_z"].append(stat["train/avg_lin_vel_z"])

            if "train/avg_base_height" in stat:
                run["metrics/avg_base_height"].append(stat["train/avg_base_height"])

            if "ep_infos" in stat:
                for key in stat["ep_infos"][0]:
                    infotensor = torch.tensor([], device="cuda:0")
                    for ep_info in stat["ep_infos"]:
                        # handle scalar and zero dimensional tensor infos
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[key]))
                    value = torch.mean(infotensor)
                    run['Episode/' + key].append(value) 

    return update


def init_wandb(cfg):
    import wandb

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True,
                                       throw_on_missing=True)
    wandb_cfg['hostname'] = platform.node()
    wandb_kwargs = cfg.logging.wandb
    wandb_tags = wandb_kwargs.get('tags', None)
    if wandb_tags is not None and isinstance(wandb_tags, str):
        wandb_kwargs['tags'] = [wandb_tags]
    if cfg.artifact is not None:
        wandb_id = cfg.artifact.split("/")[-1].split(":")[0]
        wandb_run = wandb.init(**wandb_kwargs, config=wandb_cfg, id=wandb_id, resume="must")
    else:
        wandb_run = wandb.init(**wandb_kwargs, config=wandb_cfg)
    logger.warning(f'Wandb run dir:{wandb_run.dir}')
    logger.warning(f'Project name:{wandb_run.project_name()}')
    return wandb_run


def load_class_from_path(cls_name, path):
    mod_name = 'MOD%s' % cls_name
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[cls_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)


def set_random_seed(seed=None):
    if seed is None:
        max_seed_value = np.iinfo(np.uint32).max
        min_seed_value = np.iinfo(np.uint32).min
        seed = random.randint(min_seed_value, max_seed_value)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    logger.info(f'Setting random seed to:{seed}')
    return seed


def set_print_formatting():
    """ formats numpy print """
    configs = dict(
        precision=6,
        edgeitems=30,
        linewidth=1000,
        threshold=5000,
    )
    np.set_printoptions(suppress=True,
                        formatter=None,
                        **configs)
    torch.set_printoptions(sci_mode=False, **configs)


def pathlib_file(file_name):
    if isinstance(file_name, str):
        file_name = Path(file_name)
    elif not isinstance(file_name, Path):
        raise TypeError(f'Please check the type of the filename:{file_name}')
    return file_name


def list_class_names(dir_path):
    """
    Return the mapping of class names in all files
    in dir_path to their file path.
    Args:
        dir_path (str): absolute path of the folder.
    Returns:
        dict: mapping from the class names in all python files in the
        folder to their file path.
    """
    dir_path = pathlib_file(dir_path)
    py_files = list(dir_path.rglob('*.py'))
    py_files = [f for f in py_files if f.is_file() and f.name != '__init__.py']
    cls_name_to_path = dict()
    for py_file in py_files:
        with py_file.open() as f:
            node = ast.parse(f.read())
        classes_in_file = [n for n in node.body if isinstance(n, ast.ClassDef)]
        cls_names_in_file = [c.name for c in classes_in_file]
        for cls_name in cls_names_in_file:
            cls_name_to_path[cls_name] = py_file
    return cls_name_to_path


class Tracker:
    def __init__(self, max_len):
        self.moving_average = deque([0 for _ in range(max_len)], maxlen=max_len)
        self.max_len = max_len

    def __repr__(self):
        return self.moving_average.__repr__()

    def update(self, value):
        if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            self.moving_average.extend(value.tolist())
        elif isinstance(value, Sequence):
            self.moving_average.extend(value)
        else:
            self.moving_average.append(value)

    def mean(self):
        return np.mean(self.moving_average)

    def std(self):
        return np.std(self.moving_average)

    def max(self):
        return np.max(self.moving_average)


def get_action_dim(action_space):
    if isinstance(action_space, gym.spaces.Discrete):
        act_size = action_space.n
    elif isinstance(action_space, gym.spaces.Box):
        act_size = action_space.shape[0]
    else:
        raise TypeError
    return act_size


def normalize(input, normalize_tuple):
    if normalize_tuple is not None:
        current_mean, current_var, epsilon = normalize_tuple
        y = (input - current_mean.float()) / torch.sqrt(current_var.float() + epsilon)
        y = torch.clamp(y, min=-5.0, max=5.0)
        return y
    return input


def capture_keyboard_interrupt():
    import signal
    import sys
    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


def handle_timeout(dones, info):
    timeout_key = 'TimeLimit.truncated'or 'time_outs'
    timeout_envs = None
    if timeout_key in info:
        timeout_envs = info[timeout_key]
    if timeout_envs is not None:
        dones = dones * (~timeout_envs)#bitwise not - invert sign
    return dones



def check_device(cfg):
    # sim device is always 0
    device_set = set([0, int(cfg.algo.p_learner_gpu), int(cfg.algo.v_learner_gpu)])
    if len(device_set) > cfg.available_gpus:
        assert 'Invalid CUDA device: id out of range'
    for gpu_id in device_set:
        if gpu_id >= cfg.available_gpus:
            assert f'Invalid CUDA device {gpu_id}: id out of range'
    # need more check
        
def check_hw():
    num_gpus = torch.cuda.device_count()
    print(f"Num cuda GPUs{num_gpus}")
    num_cpus = torch.cpu.device_count()
    print(f"Num cuda CPUs{num_cpus}")
    return num_gpus, num_cpus


def create_folder(folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)  
        print(f"Folder '{folder_path}' created successfully.")
    except Exception as e:
        print(f"Error creating folder '{folder_path}': {e}")

def human_readable_time(seconds):
    """
    Convert a duration in seconds to a human-readable format (years, months, days, hours, minutes).

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Human-readable string representing the duration in years, months, days, hours, and minutes.
    """
    duration = timedelta(seconds=seconds)

    years = duration.days // 365
    remaining_days = duration.days % 365
    months = remaining_days // 30
    days = remaining_days % 30
    hours = duration.seconds // 3600
    minutes = (duration.seconds % 3600) // 60

    # Format the result
    result = []
    if years > 0:
        result.append(f"{years} year{'s' if years > 1 else ''}")
    if months > 0:
        result.append(f"{months} month{'s' if months > 1 else ''}")
    if days > 0:
        result.append(f"{days} day{'s' if days > 1 else ''}")
    if hours > 0:
        result.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if minutes > 0:
        result.append(f"{minutes} minute{'s' if minutes > 1 else ''}")

    return ", ".join(result) if result else "less than a minute"


def calculate_sim_time(total_steps, simulation_time_per_step, num_environments):

    total_simulated_time = total_steps * simulation_time_per_step
    equivalent_human_time = total_simulated_time / num_environments
    # seconds_in_a_year = 60 * 60 * 24 * 365.25  # Accounting for leap years
    # years = equivalent_human_time / seconds_in_a_year
    # decades = years / 10
    # centuries = decades / 10

    return equivalent_human_time

def export_configuration(path, env, reward_config=None):
        
        config = {
        "obs": env.num_obs,
        "actions": env.num_actions,
        "dt": env.sim_params.dt,
        "control_frequency": 1.0 / (env.cfg.control.decimation * env.sim_params.dt),
        "control_mode": env.cfg.control.control_type,
        "policy_dof_names": env.dof_names,
        "default_dof_pos": env.cfg.init_state.default_joint_angles,
        "action_scale": env.cfg.control.action_scale,
        "observation_scale_dof_pos": env.cfg.normalization.obs_scales.dof_pos,
        "observation_scale_dof_vel":env.cfg.normalization.obs_scales.dof_vel,
        "clip_actions": env.cfg.normalization.clip_actions,
        "clip_observations": env.cfg.normalization.clip_observations,
        "kp": env.p_gains.cpu().numpy().tolist(),
        "kd": env.d_gains.cpu().numpy().tolist()}

        save_path = os.path.join(path, "policy_config.yaml") 
        export_config_yaml(config, save_path)

        save_path = os.path.join(path, "reward_config.yaml") 
        export_config_yaml(reward_config, save_path)


def export_config_yaml(dic_config, save_path):
    if not os.path.isfile(save_path):   
        with open(save_path, 'w') as outfile:
            yaml.dump(dic_config, outfile, default_flow_style=True)

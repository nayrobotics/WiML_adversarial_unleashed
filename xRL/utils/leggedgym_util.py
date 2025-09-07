
import os
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

#from xRL.wrappers.vector_env import VectorEnv, AMPVectorEnv
from xRL.wrappers.legged_gym_wrapper import BaseEnvWrapper, AMPEnvWrapper, LogEnvWrapper

from xRL.utils.common import export_configuration, create_folder

def create_task_env(cfg, num_envs=None, save_config = True, amp_env=False, test_env=False):
    args = get_args()

    args.task = cfg.task
    args.num_envs = num_envs if cfg.num_envs and num_envs else cfg.num_envs
    args.headless = cfg.headless
    
    args.sim_device_type = cfg.sim_device
    args.sim_device = f"cuda:{cfg.sim_device}"
    args.graphics_device_id = cfg.graphics_device_id

    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    log_path = cfg.task

    # To-do collect terrain config, command config 
    reward_config = task_registry.reward_config
    
    if save_config:
        if not os.path.isdir(log_path):
            create_folder(log_path) 
        export_configuration(log_path, env, reward_config)


    if amp_env:
        env = AMPEnvWrapper(env, env_cfg, log_path)
    elif test_env:
        env = LogEnvWrapper(env, env_cfg, log_path)
    else:
        env = BaseEnvWrapper(env, env_cfg, log_path)
    
    return env
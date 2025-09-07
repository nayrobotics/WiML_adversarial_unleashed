import isaacgym
import hydra
#import wandb
from omegaconf import DictConfig

import torch

from datetime import datetime
from loguru import logger

import xRL
from xRL.utils.leggedgym_util import create_task_env
from xRL.utils.common import capture_keyboard_interrupt

@hydra.main(config_path=xRL.LIB_PATH.joinpath('cfg').as_posix(), config_name="default")
def main(cfg: DictConfig):
    capture_keyboard_interrupt()

    cfg.num_envs = 100
    cfg.headless = False

    # Initialize Legged gym envs
    env = create_task_env(cfg, save_config = False, amp_env=False)

    action_dim = env.action_space[0]
    obs_dim = env.observation_space

    device = cfg.device

    for iter_t in range(int(cfg.max_iterations)):
        print("Interaction number:", iter_t)

        action = torch.rand((cfg.num_envs, action_dim),
                                    device=device) * 2.0 - 1.0

        next_obs, reward, done, info = env.step(action)
    print("Done")

if __name__ == '__main__':
    main()

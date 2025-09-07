
import os
import cv2
import isaacgym
import torch

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig

import xRL
from xRL.utils.common import set_random_seed
from xRL.utils.leggedgym_util import create_task_env

from xRL.utils.common import capture_keyboard_interrupt, create_folder
from xRL.utils.common import load_class_from_path
from xRL.models import model_name_to_path
from xRL.utils.common import Tracker
from xRL.utils.torch_util import RunningMeanStd 
from xRL.utils.model_util import load_model

RECORD_FRAMES = False

@hydra.main(config_path=xRL.LIB_PATH.joinpath('cfg').as_posix(), config_name="test")
def main(cfg: DictConfig):
    #set_random_seed(cfg.seed)
    capture_keyboard_interrupt()
    env = create_task_env(cfg, save_config = False, test_env=False)
    device = torch.device(cfg.device)
    obs_dim = env.observation_space#.shape
    action_dim = env.action_space[0]#.shape[0]
    
    assert cfg.artifact is not None
    act_class = load_class_from_path(cfg.algo.act_class,
                                            model_name_to_path[cfg.algo.act_class])
    
    actor = act_class(obs_dim, action_dim, cfg.algo.act_hidden_layers).to(device)
    load_model(actor, "actor", cfg)
    obs_rms = RunningMeanStd(shape=obs_dim, device=device)
    load_model(obs_rms, "obs_rms", cfg)

    return_tracker = Tracker(cfg.num_envs)
    step_tracker = Tracker(cfg.num_envs)
    current_rewards = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
    current_lengths = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)

    plot_after_inter = 100

    if RECORD_FRAMES:
        save_path = os.path.join(cfg.task, 'exported_frames')
        create_folder(save_path)

        camera_rot = 0
        camera_rot_per_sec = np.pi / 6
        img_idx = 0
        video = None

    obs = env.reset()
    for i_step in range(10*env.max_episode_length):  # run an episode
        with torch.no_grad():
            action = actor(obs_rms.normalize(obs))
        next_obs, reward, done, _ = env.step(action)
        current_rewards += reward
        current_lengths += 1
        env_done_indices = torch.where(done)[0]
        return_tracker.update(current_rewards[env_done_indices])
        step_tracker.update(current_lengths[env_done_indices])
        current_rewards[env_done_indices] = 0
        current_lengths[env_done_indices] = 0
        obs = next_obs

        #if i_step % plot_after_inter == 0:
        #    env.plot()
        
        if RECORD_FRAMES:

            # Reset camera position.
            look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
            camera_rot = (camera_rot + camera_rot_per_sec * env.dt) % (2 * np.pi)
            camera_relative_position = 1.2 * np.array([np.cos(camera_rot), np.sin(camera_rot), 0.45])
            env.set_camera(look_at + camera_relative_position, look_at)

            filename = os.path.join(save_path, f'{img_idx}.png')
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img = cv2.imread(filename)
            if video is None:
                video = cv2.VideoWriter(os.path.join(save_path, 'record.mp4'), cv2.VideoWriter_fourcc(*'MP4V'), int(1 / env.dt), (img.shape[1],img.shape[0]))
            video.write(img)
            img_idx += 1 

    video.release()
    r_exp = return_tracker.mean()
    step_exp = step_tracker.mean()
    logger.warning(f"Cumulative return: {r_exp}, Episode length: {step_exp}")


if __name__ == '__main__':
    main()
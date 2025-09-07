from itertools import count
import isaacgym
import hydra
#import wandb
from omegaconf import DictConfig
import os
import time
import xRL
from xRL.algo import alg_name_to_path
from xRL.replay.replay_buffer import ReplayBuffer
from xRL.utils.common import manage_tracking
from xRL.utils.common import load_class_from_path, human_readable_time, create_folder
from xRL.utils.evaluator import Evaluator
from xRL.utils.leggedgym_util import create_task_env
from xRL.utils.common import capture_keyboard_interrupt
from xRL.utils.model_util import load_model, save_model

from datetime import datetime
from loguru import logger

@hydra.main(config_path=xRL.LIB_PATH.joinpath('cfg').as_posix(), config_name="default")
def main(cfg: DictConfig):
    capture_keyboard_interrupt()

    # Initialize tracking: neptune or wandb (not implemented)
    update_run = manage_tracking(cfg.tracking_training, params=cfg)

    is_amp = cfg.with_imitation_amp 
    buffer_is_full = False
    buffer_is_full_it = 0
    
    log_dir = cfg.task
    create_folder(log_dir)

    # Initialize Legged gym envs
    env = create_task_env(cfg, amp_env=is_amp)

    algo_name = cfg.algo.name
    print("Initializing agent", algo_name)


    if 'Agent' not in algo_name:
        algo_name = 'Agent' + algo_name
    agent_class = load_class_from_path(algo_name, alg_name_to_path[algo_name])
    agent = agent_class(env=env, cfg=cfg)

    if cfg.artifact is not None:
        # load trained model - To-Do
        load_model(agent.actor, "actor", cfg)
        load_model(agent.critic, "critic", cfg)
        if cfg.algo.obs_norm:
            load_model(agent.obs_rms, "obs_rms", cfg)

    global_steps = 0
    tot_learning_steps = 0
    finished =  False

    if cfg.evaluator:
        evaluator = Evaluator(cfg=cfg, log_dir=log_dir)

    start_time = time.time()

    agent.reset_agent()
    is_off_policy = cfg.algo.name != 'PPO'

    if is_off_policy:
        memory = ReplayBuffer(capacity=int(cfg.algo.memory_size),
                              obs_dim=agent.obs_dim,
                              action_dim=agent.action_dim,
                              amp_dim=agent.amp_obs_dim,
                              device=cfg.rl_buffer_device)
        trajectory, steps, ep_infos = agent.explore_env(env, cfg.algo.warm_up, random=True)
        
        memory.add_to_buffer(trajectory)
        global_steps += steps

        # Upload policy config and reward config from legged_gym
        rw_config_path = f"{log_dir}/reward_config.yaml" 
        update_run(cfg.tracking_training, reward_config=rw_config_path)

        policy_config_path = f"{log_dir}/policy_config.yaml" 
        update_run(cfg.tracking_training, policy_config=policy_config_path)

    for iter_t in count():
        print("Interaction number:", iter_t)
        trajectory, steps, ep_infos = agent.explore_env(env, cfg.algo.num_interac_per_env, random=False)
        global_steps += steps
        

        if is_off_policy:
            memory.add_to_buffer(trajectory)
            log_info = agent.update_net(memory)

            if memory.if_full and buffer_is_full is False:
                buffer_is_full = True
                buffer_is_full_it = iter_t
        else:
            log_info = agent.update_net(trajectory)

        print("Buffer is full:", buffer_is_full)

        if iter_t % cfg.algo.log_freq == 0:

            delta_time = time.time() - start_time
            ct = human_readable_time(delta_time)

            tot_learning_steps += cfg.algo.num_interac_per_env
            #num_envs = env.num_envs
            #sim_time_step = env.sim_time_step
            
            if is_off_policy:
                log_info["buffer_is_full_it"] = buffer_is_full_it


            log_info['global_steps'] = global_steps
            log_info['episodes'] = iter_t
            log_info["ep_infos"] = ep_infos
            log_info["ct"] = ct
            log_info["ct_seconds"] = delta_time

            log_info["train/avg_lin_vel_x"] = env.base_lin_vel_x
            log_info["train/avg_lin_vel_y"] = env.base_lin_vel_y
            log_info["train/avg_lin_vel_z"] = env.base_lin_vel_z
            log_info["train/avg_base_height"] = env.base_height
            

            update_run(cfg.tracking_training, log_info)


            logger.info(f"['Episodes']:{iter_t:12.2e}"
                        #f"['Steps']:{global_steps:12.2e}"
                         #f"['Time']:{delta_time:>12.1f}"
                         f"['train/return']:{log_info['train/return']:12.2f}")                       
            #             f"['train/critic_loss']:{log_info['train/critic_loss']:12.2f}"
            #             f"['train/actor_loss']:{log_info['train/actor_loss']:12.2f}")
        

        if cfg.evaluator:
            if evaluator.parent.poll():
                return_dict = evaluator.parent.recv()
                update_run(cfg.tracking_training, return_dict)

            if iter_t % cfg.algo.eval_freq == 0:
                logger.info(f"['Steps']: {global_steps:12.2e}"
                            f"['Time']: {time.time() - start_time:>12.1f}"
                            f"['train/return']:{log_info['train/return']:12.2f}"                       
                            f"['train/critic_loss']:{log_info['train/critic_loss']:12.2f}"
                            f"['train/actor_loss']:{log_info['train/actor_loss']:12.2f}")
                evaluator.eval_policy(agent.actor, agent.critic, normalizer=agent.obs_rms,
                                    step=global_steps)
        
        if cfg.max_step is not None and global_steps > cfg.max_step:
                print("Max step reached:", cfg.max_step)
                finished = True
        elif cfg.max_time is not None and (time.time() - start_time) > cfg.max_time:
            print("Max time reached:", cfg.max_time)
            finished = True
        
        elif cfg.max_iterations is not None and iter_t == (cfg.max_iterations - 1):
            print("Max number of interations reached:", cfg.max_iterations)
            finished = True

        if iter_t % cfg.algo.save_interval == 0 or finished:
            save_path = f"{log_dir}/_{iter_t}_model.pth"
            if finished:
                save_path = f"{log_dir}/model.pth"
            save_model(path=save_path,
                                actor=agent.actor.state_dict(),
                                critic=agent.critic.state_dict(),
                                rms=agent.obs_rms.get_states() if cfg.algo.obs_norm else None)
            
            update_run(cfg.tracking_training, log_info, save_path)

        if finished:
            break


if __name__ == '__main__':
    main()

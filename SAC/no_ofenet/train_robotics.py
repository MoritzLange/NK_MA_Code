#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils

# import dmc2gym
import hydra
import gym

# from network import OFENet
import wandb

class ObservationWrapper(gym.ObservationWrapper):
    """ Observation wrapper that takes only the observation out of the robot arm FetchSlideDense task.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
                    -np.inf, np.inf, shape=(31,), dtype="float32"
                )

    def observation(self, obs):
        # modify obs
        return np.concatenate([obs["observation"], obs["achieved_goal"], obs["desired_goal"]])

    def __getattr__(self, name):
        return getattr(self.env, name)

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        cfg.agent.params.env_name = cfg.env
        cfg.agent.params.aux_task = cfg.aux_task
        cfg.agent.params.total_units = cfg.total_units
        self.env = ObservationWrapper(gym.make(cfg.env))

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.step = 0
        self.avg_rew = 0
        self.success_rate = 0

    @property
    def evaluate(self):

        average_episode_reward = 0
        count = 0
        success_counter = 0
        evaluate_buffer = ReplayBuffer(self.env.observation_space.shape,
                                       self.env.action_space.shape,
                                       int(self.cfg.replay_buffer_capacity),
                                       self.device)
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            success=False
            for _ in range(50):
                state = obs
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.env.step(action)
                if info["is_success"]:
                    success = True
                evaluate_buffer.add(state, action, reward, obs, done, done)
                episode_reward += reward
                count += 1

            average_episode_reward += episode_reward
            if success:
                success_counter += 1
            
        self.success_rate = success_counter/self.cfg.num_eval_episodes
        average_episode_reward /= self.cfg.num_eval_episodes

        self.avg_rew = average_episode_reward
        print("---------------------------------------")
        print(f"Evaluation over {self.cfg.num_eval_episodes} episodes: Avg. reward is {average_episode_reward:.3f} and success rate is {self.success_rate:.3f}")
        print("---------------------------------------")

    def run(self):
        episode, episode_reward, done = 0, 0, True
        batch_size = 256
        episode_step = 0
        obs = self.env.reset()
        
        if self.cfg.wandb_name != "unnamed":
            config = {
                "env_name": self.cfg.env,
                "aux_task": self.cfg.aux_task,
                "max_timesteps": self.cfg.num_train_steps,
                "start_timesteps": self.cfg.num_seed_steps,
                "num_pretrain": self.cfg.num_pretrain,
                "total_units": self.cfg.total_units,
                "seed": self.cfg.seed,
                "batch_size": batch_size,
            }
            wandb.init(project=self.cfg.wandb_name, entity=self.cfg.wandb_entity, config={**config, **vars(self.cfg)})

        while self.step < self.cfg.num_train_steps:
            if self.step >= 0 and self.step % self.cfg.eval_frequency == 0:
                self.evaluate
                print(
                    f"Total T: {self.step + 1} Episode Num: {episode + 1} Episode T: {episode_step} Reward: {episode_reward:.3f}")
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
            
            if self.cfg.wandb_name != "unnamed":
                if self.step % self.cfg.eval_frequency == 0:
                    wandb_logs = {
                        "Reward": self.avg_rew,
                        "Success rate": self.success_rate,
                        "Step": self.step,

                    }
                    wandb.log(wandb_logs)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            # collecting samples is not needed, if ofenet was pretrained.
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()

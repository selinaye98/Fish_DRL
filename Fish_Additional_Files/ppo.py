import gym_fish
import gym_fish_test
import gymnasium as gym
import numpy as np
import torch
import os

from torch import nn
from torch.distributions import Independent, Normal
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net, DataParallelNet
from tianshou.utils.net.discrete import Actor, Critic

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

import warnings


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_envs = gym.make('FISH-v0')
    test_envs = gym.make('FISH-TEST-v0')
    # net is the shared head of the actor and the critic
    net = Net(train_envs.observation_space.shape, hidden_sizes=[128, 128], activation=nn.Tanh, device=device)
    actor = ActorProb(net, train_envs.action_space.shape,max_action=train_envs.action_space.high[0], device=device).to(device)
    # actor = DataParallelNet(ActorProb(net, train_envs.action_space.shape, max_action=train_envs.action_space.high[0], device=None).to(device))

    net_c = Net(train_envs.observation_space.shape, hidden_sizes=[128, 128], activation=nn.Tanh, device=device)
    critic = Critic(net_c, device=device).to(device)
    #critic = DataParallelNet(Critic(net_c, device=None).to(device))
    actor_critic = ActorCritic(actor, critic)

    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    # optimizer of the actor and the critic
    optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / 300)

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=0.99,
        eps_clip=0.2,
        dual_clip=None,
        value_clip=True,
        advantage_normalization=True,
        recompute_advantage=False,
        vf_coef=0.25,
        ent_coef=0.0,
        max_grad_norm=None,
        gae_lambda=0.95,
        reward_normalization=False,
        max_batchsize=1024,
        action_scaling=True,
        action_bound_method='tanh',
        action_space=train_envs.action_space,
        lr_scheduler=lr_scheduler,
        deterministic_eval=True
    )

    train_collector = Collector(policy, train_envs, VectorReplayBuffer(400, 1))
    test_collector = Collector(policy, test_envs)

    log_path = os.path.join('log', "ppo", '7')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, save_interval=1)

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save(
            {
                "model": policy.state_dict(),
                "optim": optim.state_dict(),
            }, ckpt_path
        )
        return ckpt_path

    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=60,
        step_per_epoch=1000,
        repeat_per_collect=10,
        episode_per_test=1,
        batch_size=128,
        step_per_collect=200,
        stop_fn=lambda mean_reward: mean_reward >= 1000,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
    )

    # Let's watch its performance!
    policy.eval()
    result_test = test_collector.collect(n_episode=1, render=False)
    print("Final reward: {}, length: {}".format(result["rews"].mean(), result["lens"].mean()))
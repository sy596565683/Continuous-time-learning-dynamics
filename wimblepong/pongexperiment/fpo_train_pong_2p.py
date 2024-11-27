from spinup import fpo_tf1 as fpo
import gym
import wimblepong
import argparse
from spinup.utils.mpi_tools import mpi_fork
from spinup.utils.run_utils import setup_logger_kwargs
import spinup.algos.tf1.ppo.core as core
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


parser = argparse.ArgumentParser()
parser.add_argument('--hid', type=int, default=64)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=100)
parser.add_argument('--cpu', type=int, default=6)
parser.add_argument('--steps', type=int, default=4000)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--max_ep_len', type=int, default=600)  # smaller than steps/cpu
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--exp_name', type=str, default='fpo_clip_entropy(0.01)_replay(1)_pong_2p')
args = parser.parse_args()

mpi_fork(args.cpu)  # run parallel code with mpi

logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

fpo(lambda: gym.make("WimblepongMultiplayer-v0"),
    lambda: gym.make("WimblepongSimpleAI-v0"),
    actor_critic=core.mlp_actor_critic, clip_ratio=0.2,
    ent_coef=0.01,
    replay_epochs=1,
    ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
    seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
    max_ep_len=args.max_ep_len, save_freq=args.save_freq,
    logger_kwargs=logger_kwargs)

# fpo(lambda: gym.make("WimblepongMultiplayer-v0"),
#     lambda: gym.make("WimblepongSimpleAI-v0"),
#     actor_critic=core.mlp_actor_critic, clip_ratio=10000,
#     ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
#     seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
#     max_ep_len=args.max_ep_len, save_freq=args.save_freq,
#     logger_kwargs=logger_kwargs)

# fpo(lambda: gym.make("WimblepongMultiplayer-v0"),
#     lambda: gym.make("WimblepongSimpleAI-v0"),
#     actor_critic=core.mlp_actor_critic, clip_ratio=10000,
#     ent_coef=0.01,
#     ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
#     seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
#     max_ep_len=args.max_ep_len, save_freq=args.save_freq,
#     logger_kwargs=logger_kwargs)

# fpo(lambda: gym.make("WimblepongMultiplayer-v0"),
#     lambda: gym.make("WimblepongSimpleAI-v0"),
#     actor_critic=core.mlp_actor_critic, clip_ratio=10000,
#     ent_coef=0,
#     replay_epochs=100,
#     ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
#     seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
#     max_ep_len=args.max_ep_len, save_freq=args.save_freq,
#     logger_kwargs=logger_kwargs)

import gym
import wimblepong
import argparse
from spinup.utils.test_policy import load_policy_and_env, run_policy


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--housekeeping", action="store_true", help="Plot, player and ball positions and velocities at the end of each episode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongSimpleAI-v0")
#env.unwrapped.scale = args.scale
#env.unwrapped.fps = args.fps


# _, get_action = load_policy_and_env('../data/ppo_pong_single/ppo_pong_single_s0/')
_, get_action = load_policy_and_env('../data/fpo_clip_entropy(0.1)_pong_2p/fpo_clip_entropy(0.1)_pong_2p_s200_backup/', itr='last')
env.set_names(p1="EPO")
run_policy(env, get_action)

import gym
import wimblepong
import time
import numpy as np
from spinup.utils.test_policy import load_policy_and_env, run_policy
import tensorflow as tf
from wimblepong.simple_ai import SimpleAi
import imageio

# Make the environment
#env = gym.make("WimblepongSimpleAI-v0")
env = gym.make("WimblepongMultiplayer-v0")

graph1 = tf.Graph()
with graph1.as_default():
    _, get_p1_action = load_policy_and_env('../data/fpo_clip_entropy(0.1)_pong_2p/fpo_clip_entropy(0.1)_pong_2p_s200_backup/', itr=399)
graph2 = tf.Graph()
env.set_names(p1="EPO")

# with graph2.as_default(): _, get_p2_action = load_policy_and_env('../data/fpo_clip_entropy(
# 0)_pong_2p/fpo_clip_entropy(0)_pong_2p_s300_backup/', itr=399)
simple_ai = SimpleAi(env, 2)
get_p2_action = simple_ai.get_action
env.set_names(p2="SimpleAi")

#run_policy(env, get_action)

gif_frames = []
gif_length = 3*60*env.fps
gif_count = 0
gif_save = False

o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0.0, 0
test_rounds, Awins, Bwins, draws = 0, 0, 0, 0
while test_rounds < 1000:
    if True:
        frame = env.render()
        if gif_count < gif_length:
            gif_count += 1
            gif_frames.append(frame)
        elif gif_save is False:
            gif_save = True
            imageio.mimsave('pong_iter_399.gif', gif_frames, format='gif', fps=env.fps)
            print('gif saved')


    a1 = get_p1_action(o[0])
    a2 = get_p2_action(o[1])
    o2, r, d, _ = env.step([a1, a2])
    ep_ret += r
    ep_len += 1

    o = o2

    if np.mean(d) or (ep_len == 2000):
        #print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret[0], ep_len))
        if r[0] > 0:
            Awins += 1
        elif r[0] < 0:
            Bwins += 1
        else:
            draws += 1
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0.0, 0

        test_rounds += 1
        if test_rounds % 10 == 0:
            print('Test rounds %d \t Awins %.3f \t Bwins %.3f \t draws %.3f' %
                  (test_rounds, Awins / test_rounds, Bwins / test_rounds, draws / test_rounds))

print('Test rounds %d \t Awins %.3f \t Bwins %.3f \t draws %.3f'%
      (test_rounds, Awins/test_rounds, Bwins/test_rounds, draws/test_rounds))
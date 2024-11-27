import gym
import wimblepong
from wimblepong.simple_ai import SimpleAi
import numpy as np
from spinup.utils.test_policy import load_policy_and_env, run_policy
import time
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import argparse
import os
import tensorflow as tf


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def calculate_play_result(env, policy1, policy2, rounds=100, render=False):
    A_wins, B_wins, draws = 0, 0, 0
    o, r, d, ep_len = env.reset(), 0, False, 0
    test_round = 0
    while test_round < rounds:
        if render:
            env.render()
            time.sleep(1e-3)

        a1 = policy1(o[0])
        a2 = policy2(o[1])
        o2, r, d, _ = env.step([a1, a2])
        ep_len += 1

        o = o2
        if np.mean(d) or (ep_len == 5000):
            # print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret[0], ep_len))
            if r[0] > 0:
                A_wins += 1
            elif r[0] < 0:
                B_wins += 1
            else:
                draws += 1

            o, r, d, ep_len = env.reset(), 0, False, 0
            test_round += 1

    return [A_wins/test_round, B_wins/test_round, draws/test_round]


parser = argparse.ArgumentParser()
parser.add_argument('--seed', '-s', type=int, default=100)
parser.add_argument('--cpu', type=int, default=20)
parser.add_argument('--rounds', type=int, default=1000)
# parser.add_argument('--exp_name', type=str, default='compare_fpo_clip_entropy(0)_player')
# parser.add_argument('--exp_name', type=str, default='compare_fpo_clip_entropy(0.03)_player')
# parser.add_argument('--exp_name', type=str, default='compare_fpo_clip_entropy(0.03)_replay(1)_player')
# parser.add_argument('--exp_name', type=str, default='compare_fpo_clip_entropy(0.03)_replay(10)_player')
# parser.add_argument('--exp_name', type=str, default='compare_fpo_clip_entropy(0.03)_replay(100)_player')
parser.add_argument('--exp_name', type=str, default='compare_fpo_clip_entropy(0.1)_replay(100)_player')
# parser.add_argument('--exp_name', type=str, default='compare_fpo_clip_entropy(0.1)_player')
args = parser.parse_args()

mpi_fork(args.cpu)  # run parallel code with mpi

logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
logger = EpochLogger(**logger_kwargs)
#logger.save_config(locals())
local_rounds = int(args.rounds / num_procs())

# Make the environment
env = gym.make("WimblepongMultiplayer-v0")

start_time = time.time()

# player list
# player_path = '../data/fpo_clip_entropy(0.03)_pong_2p/fpo_clip_entropy(0.03)_pong_2p_s300_backup/'
# player_path = '../data/fpo_clip_entropy(0.03)_replay(1)_pong_2p/fpo_clip_entropy(0.03)_replay(1)_pong_2p_s300_backup/'
# player_path = '../data/fpo_clip_entropy(0.03)_replay(10)_pong_2p/fpo_clip_entropy(0.03)_replay(10)_pong_2p_s300_backup/'
# player_path = '../data/fpo_clip_entropy(0.03)_replay(100)_pong_2p/fpo_clip_entropy(0.03)_replay(100)_pong_2p_s300_backup/'
player_path = '../data/fpo_clip_entropy(0.1)_replay(100)_pong_2p/fpo_clip_entropy(0.1)_replay(100)_pong_2p_s100_backup/'
# player_path = '../data/fpo_clip_entropy(0.1)_pong_2p/fpo_clip_entropy(0.1)_pong_2p_s100_backup/'
player_list = [i * 20 for i in range(20)] + ['last'] + ['simpleai']
#player_list = [0,100,200] + ['last'] + ['simpleai']

# record matching results
match_payoff = np.zeros((len(player_list), len(player_list)))

get_actions = [None for i in range(len(player_list))]
for i in range(len(player_list)):
    if player_list[i] is 'simpleai':
        simple_ai = SimpleAi(env, 1)
        get_actions[i] = simple_ai.get_action
    else:
        graph = tf.Graph()
        with graph.as_default():
            _, get_actions[i] = load_policy_and_env(player_path, itr=player_list[i])

    a1 = get_actions[i](np.array([0,0,0,0,0,0]))

epoch = 0
for i in range(len(player_list)):
    for j in range(i,len(player_list)):
        [Awin_rate1, Bwin_rate1, draw_rate1] = calculate_play_result(
            env, get_actions[i], get_actions[j], rounds=int(local_rounds / 2))
        [Awin_rate2, Bwin_rate2, draw_rate2] = calculate_play_result(
            env, get_actions[j], get_actions[i], rounds=int(local_rounds / 2))

        logger.save_state({'match': [i, j]}, epoch)
        logger.store(AWinRate=(Awin_rate1 + Bwin_rate2) / 2,
                     BWinRate=(Bwin_rate1 + Awin_rate2) / 2,
                     DrawRate=(draw_rate1 + draw_rate2) / 2)

        avg_Awin_rate = logger.get_stats('AWinRate')[0]
        avg_Bwin_rate = logger.get_stats('BWinRate')[0]
        avg_draw_rate = logger.get_stats('DrawRate')[0]
        if proc_id() == 0:
            print('player1=%s, player2=%s' % (str(i), str(j)))
            print('Test rounds %d \t Awins %.3f \t Bwins %.3f \t draws %.3f' %
                  (args.rounds, avg_Awin_rate,
                   avg_Bwin_rate, avg_draw_rate))

        match_payoff[i, j] = avg_Awin_rate - avg_Bwin_rate
        # print(match_payoff[i, j])

        logger.log_tabular('Match', str(player_list[i]) + '_vs_' + str(player_list[j]))
        logger.log_tabular('AWinRate', average_only=True)
        logger.log_tabular('BWinRate', average_only=True)
        logger.log_tabular('DrawRate', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

    epoch += 1


match_payoff = match_payoff - match_payoff.T

if proc_id() == 0:
    print(match_payoff)
    np.save(player_path + 'match_payoff', match_payoff)


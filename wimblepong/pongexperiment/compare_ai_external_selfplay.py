import gym
import wimblepong
from wimblepong.simple_ai import SimpleAi
import numpy as np
from spinup.utils.test_policy import load_policy_and_env, load_fic_policy_and_env, run_policy
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
parser.add_argument('--cpu', type=int, default=15)
parser.add_argument('--rounds', type=int, default=1000)
# parser.add_argument('--exp_name', type=str, default='compare_fpo_agents')
parser.add_argument('--exp_name', type=str, default='compare_selfplay_diff_agents')
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
path_list_self = ['../data/fpo_clip_replay(1)_pong_2p/fpo_clip_replay(1)_pong_2p_s{}_backup/',
                  '../data/fpo_clip_entropy(0.01)_replay(1)_pong_2p/fpo_clip_entropy(0.01)_replay(1)_pong_2p_s{}_backup/',]

path_list_other = ['../data/fpo_clip_entropy(0.1)_pong_2p/fpo_clip_entropy(0.1)_pong_2p_s{}_backup/',

                   '../data/fpo_clip_replay(1)_pong_2p/fpo_clip_replay(1)_pong_2p_s{}_backup/',
                   '../data/fpo_clip_entropy(0.003)_replay(1)_pong_2p/fpo_clip_entropy(0.003)_replay(1)_pong_2p_s{}_backup/',
                   '../data/fpo_clip_entropy(0.01)_replay(1)_pong_2p/fpo_clip_entropy(0.01)_replay(1)_pong_2p_s{}_backup/',
                   '../data/fpo_clip_entropy(0.03)_replay(1)_pong_2p/fpo_clip_entropy(0.03)_replay(1)_pong_2p_s{}_backup/',
                   '../data/fpo_clip_entropy(0.1)_replay(1)_pong_2p/fpo_clip_entropy(0.1)_replay(1)_pong_2p_s{}_backup/',
                   '../data/fpo_clip_entropy(0.3)_replay(1)_pong_2p/fpo_clip_entropy(0.3)_replay(1)_pong_2p_s{}_backup/',

                   '../data/nfsp_pong_2p/nfsp_pong_2p_s{}_backup/',
                   '../data/nfsp_entropy(0.003)_pong_2p/nfsp_entropy(0.003)_pong_2p_s{}_backup/',
                   '../data/nfsp_entropy(0.01)_pong_2p/nfsp_entropy(0.01)_pong_2p_s{}_backup/',
                   '../data/nfsp_entropy(0.03)_pong_2p/nfsp_entropy(0.03)_pong_2p_s{}_backup/',
                   '../data/nfsp_entropy(0.1)_pong_2p/nfsp_entropy(0.1)_pong_2p_s{}_backup/',
                   '../data/nfsp_entropy(0.3)_pong_2p/nfsp_entropy(0.3)_pong_2p_s{}_backup/',

                   '../data/ppo_pong_simpleai/ppo_pong_simpleai_s{}_backup/',
                   '../data/ppo_entropy(0.003)_pong_simpleai/ppo_entropy(0.003)_pong_simpleai_s{}_backup/',
                   '../data/ppo_entropy(0.01)_pong_simpleai/ppo_entropy(0.01)_pong_simpleai_s{}_backup/',
                   '../data/ppo_entropy(0.03)_pong_simpleai/ppo_entropy(0.03)_pong_simpleai_s{}_backup/',
                   '../data/ppo_entropy(0.1)_pong_simpleai/ppo_entropy(0.1)_pong_simpleai_s{}_backup/',
                   '../data/ppo_entropy(0.3)_pong_simpleai/ppo_entropy(0.3)_pong_simpleai_s{}_backup/',

                   'simpleai'
]

get_actions_self = []
players_self = []
for i in range(len(path_list_self)):
    if path_list_self[i] is 'simpleai':
        get_actions_self.append([])
        for s in range(3):
            simple_ai = SimpleAi(env, 1)
            get_actions_self[-1].append(simple_ai.get_action)
        players_self.append(path_list_self[i])
    else:
        get_actions_self.append([])
        for s in range(3):
            loc = path_list_self[i].format((s + 1) * 100)
            graph = tf.Graph()
            with graph.as_default():
                _, op = load_policy_and_env(loc, itr='last')
                get_actions_self[-1].append(op)
        players_self.append(path_list_self[i])
        if 'nfsp' in path_list_self[i]:
            get_actions_self.append([])
            for s in range(3):
                loc = path_list_self[i].format((s + 1) * 100)
                graph = tf.Graph()
                with graph.as_default():
                    _, op = load_fic_policy_and_env(loc, itr='last')
                    get_actions_self[-1].append(op)
            players_self.append(path_list_self[i]+'fic')

for i in range(len(players_self)):
    for s in range(3):
        a1 = get_actions_self[i][s](np.array([0,0,0,0,0,0]))
        
        
get_actions_other = []
players_other = []
for i in range(len(path_list_other)):
    if path_list_other[i] is 'simpleai':
        get_actions_other.append([])
        for s in range(3):
            simple_ai = SimpleAi(env, 1)
            get_actions_other[-1].append(simple_ai.get_action)
        players_other.append(path_list_other[i])
    else:
        get_actions_other.append([])
        for s in range(3):
            loc = path_list_other[i].format((s + 1) * 100)
            graph = tf.Graph()
            with graph.as_default():
                _, op = load_policy_and_env(loc, itr='last')
                get_actions_other[-1].append(op)
        players_other.append(path_list_other[i])
        if 'nfsp' in path_list_other[i]:
            get_actions_other.append([])
            for s in range(3):
                loc = path_list_other[i].format((s + 1) * 100)
                graph = tf.Graph()
                with graph.as_default():
                    _, op = load_fic_policy_and_env(loc, itr='last')
                    get_actions_other[-1].append(op)
            players_other.append(path_list_other[i]+'fic')

for i in range(len(players_other)):
    for s in range(3):
        a1 = get_actions_other[i][s](np.array([0,0,0,0,0,0]))

# record matching results
match_payoff = np.zeros((len(players_self), len(players_other), 3, 3))

epoch = 0
for i in range(len(players_self)):
    for j in range(len(players_other)):
        for s1 in range(3):
            for s2 in range(3):
                [Awin_rate1, Bwin_rate1, draw_rate1] = calculate_play_result(
                    env, get_actions_self[i][s1], get_actions_other[j][s2], rounds=int(local_rounds / 2))
                [Awin_rate2, Bwin_rate2, draw_rate2] = calculate_play_result(
                    env, get_actions_other[j][s2], get_actions_self[i][s1], rounds=int(local_rounds / 2))

                logger.save_state({'match': [i,j,s1,s2]}, epoch)
                logger.store(AWinRate=(Awin_rate1+Bwin_rate2)/2,
                             BWinRate=(Bwin_rate1+Awin_rate2)/2,
                             DrawRate=(draw_rate1+draw_rate2)/2)

                avg_Awin_rate = logger.get_stats('AWinRate')[0]
                avg_Bwin_rate = logger.get_stats('BWinRate')[0]
                avg_draw_rate = logger.get_stats('DrawRate')[0]
                if proc_id() == 0:
                    print('player1=%s, player2=%s, s1=%s, s2=%s' % (str(i), str(j),
                                                                    str((s1+1)*100), str((s2+1)*100)))
                    print('Test rounds %d \t Awins %.3f \t Bwins %.3f \t draws %.3f' %
                          (args.rounds, avg_Awin_rate, avg_Bwin_rate, avg_draw_rate))

                match_payoff[i, j, s1, s2] = avg_Awin_rate - avg_Bwin_rate
                # print(match_payoff[i, j])

                logger.log_tabular('Match', players_self[i].format((s1+1)*100) +
                                   '_vs_' + players_other[j].format((s2+1)*100))
                logger.log_tabular('AWinRate', average_only=True)
                logger.log_tabular('BWinRate', average_only=True)
                logger.log_tabular('DrawRate', average_only=True)
                logger.log_tabular('Time', time.time()-start_time)
                logger.dump_tabular()

                epoch += 1


if proc_id() == 0:
    print(match_payoff)
    np.save('../data/' + args.exp_name + '/match_payoff', match_payoff)
    np.save('../data/' + args.exp_name + '/players_self', players_self)
    np.save('../data/' + args.exp_name + '/players_other', players_other)


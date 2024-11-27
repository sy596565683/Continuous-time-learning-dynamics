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


base_name = 'fpo_clip_entropy(0.1)_pong_2p'
# base_name = 'fpo_clip800_entropy(0.1)_pong_2p'

parser = argparse.ArgumentParser()
parser.add_argument('--seed', '-s', type=int, default=100)
parser.add_argument('--cpu', type=int, default=15)
parser.add_argument('--rounds', type=int, default=180)
# parser.add_argument('--exp_name', type=str, default='compare_baseline_'+base_name)
# parser.add_argument('--exp_name', type=str, default='compare_baseline_epoch_'+base_name)
parser.add_argument('--exp_name', type=str, default='compare_baseline_epoch10_'+base_name)
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
path_baseline = '../data/'+base_name+'/'+base_name+'_s{}_backup/'

other_base_names = ['fpo_clip_replay(1)_pong_2p',
                    'fpo_clip_entropy(0.003)_replay(1)_pong_2p',
                    'fpo_clip_entropy(0.01)_replay(1)_pong_2p',
                    'fpo_clip_entropy(0.03)_replay(1)_pong_2p',
                    'fpo_clip_entropy(0.1)_replay(1)_pong_2p',
                    'fpo_clip_entropy(0.3)_replay(1)_pong_2p',

                    'nfsp_pong_2p',
                    'nfsp_entropy(0.003)_pong_2p',
                    'nfsp_entropy(0.01)_pong_2p',
                    'nfsp_entropy(0.03)_pong_2p',
                    'nfsp_entropy(0.1)_pong_2p',
                    'nfsp_entropy(0.3)_pong_2p',

                    'ppo_pong_simpleai',
                    'ppo_entropy(0.003)_pong_simpleai',
                    'ppo_entropy(0.01)_pong_simpleai',
                    'ppo_entropy(0.03)_pong_simpleai',
                    'ppo_entropy(0.1)_pong_simpleai',
                    'ppo_entropy(0.3)_pong_simpleai',

                    'simpleai',
               ]

player_self = path_baseline

for other in range(len(other_base_names)):
    if 'nfsp' in other_base_names[other]:
        other_names = [other_base_names[other], other_base_names[other]+'_fic']
    else:
        other_names = [other_base_names[other]]
    for rpt in range(len(other_names)):
        path_other = '../data/' + other_base_names[other] + '/' + other_base_names[other] + '_s{}_backup/'
        other_name = other_names[rpt]
        match_payoff = np.zeros((41, 3, 3))
        epoch = 0
        for i in range(0, 401, 10):
            if i == 400:
                itr = 'last'
            else:
                itr = i
            get_actions_self = []
            for s in range(3):
                loc = path_baseline.format((s + 1) * 100)
                graph = tf.Graph()
                with graph.as_default():
                    _, op = load_policy_and_env(loc, itr=itr)
                    get_actions_self.append(op)
            
            get_actions_other = []
            for s in range(3):
                if other_name is 'simpleai':
                    simple_ai = SimpleAi(env, 1)
                    get_actions_other.append(simple_ai.get_action)
                else:
                    loc = path_other.format((s + 1) * 100)
                    graph = tf.Graph()
                    if 'fic' in other_name:
                        with graph.as_default():
                            _, op = load_fic_policy_and_env(loc, itr=itr)
                            get_actions_other.append(op)
                    else:
                        with graph.as_default():
                            _, op = load_policy_and_env(loc, itr=itr)
                            get_actions_other.append(op)
    
            for s1 in range(3):
                for s2 in range(3):
                    [Awin_rate1, Bwin_rate1, draw_rate1] = calculate_play_result(
                        env, get_actions_self[s1], get_actions_other[s2], rounds=int(local_rounds / 2))
                    [Awin_rate2, Bwin_rate2, draw_rate2] = calculate_play_result(
                        env, get_actions_other[s2], get_actions_self[s1], rounds=int(local_rounds / 2))
    
                    logger.save_state({'match': [epoch, s1, s2]}, epoch)
                    logger.store(AWinRate=(Awin_rate1 + Bwin_rate2) / 2,
                                 BWinRate=(Bwin_rate1 + Awin_rate2) / 2,
                                 DrawRate=(draw_rate1 + draw_rate2) / 2)
    
                    avg_Awin_rate = logger.get_stats('AWinRate')[0]
                    avg_Bwin_rate = logger.get_stats('BWinRate')[0]
                    avg_draw_rate = logger.get_stats('DrawRate')[0]
                    if proc_id() == 0:
                        print('player2=%s, iter=%s, s1=%s, s2=%s' % (other_name, str(i),
                                                                        str((s1 + 1) * 100), str((s2 + 1) * 100)))
                        print('Test rounds %d \t Awins %.3f \t Bwins %.3f \t draws %.3f' %
                              (args.rounds, avg_Awin_rate, avg_Bwin_rate, avg_draw_rate))
    
                    match_payoff[epoch, s1, s2] = avg_Awin_rate - avg_Bwin_rate
                    # print(match_payoff[i, j])
    
                    logger.log_tabular('Match', base_name + str((s1 + 1) * 100) + '_iter_' + str(i) +
                                       '_vs_' + other_name + str((s2 + 1) * 100) + '_iter_' + str(i))
                    logger.log_tabular('AWinRate', average_only=True)
                    logger.log_tabular('BWinRate', average_only=True)
                    logger.log_tabular('DrawRate', average_only=True)
                    logger.log_tabular('Time', time.time() - start_time)
                    logger.dump_tabular()
    
            epoch += 1
    
        if proc_id() == 0:
            print(match_payoff)
            np.save('../data/' + args.exp_name + '/match_payoff_' + other_name, match_payoff)


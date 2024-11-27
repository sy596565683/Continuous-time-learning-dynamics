import matplotlib.pyplot as plt
import image_annotated_heatmap as iah
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['pdf.fonttype'] = 42

other_names = ['fpo_clip_replay(1)_pong_2p',
               'fpo_clip_entropy(0.003)_replay(1)_pong_2p',
               'fpo_clip_entropy(0.01)_replay(1)_pong_2p',
               'fpo_clip_entropy(0.03)_replay(1)_pong_2p',
               'fpo_clip_entropy(0.1)_replay(1)_pong_2p',
               'fpo_clip_entropy(0.3)_replay(1)_pong_2p',

               'nfsp_pong_2p_fic',
               'nfsp_entropy(0.003)_pong_2p_fic',
               'nfsp_entropy(0.01)_pong_2p_fic',
               'nfsp_entropy(0.03)_pong_2p_fic',
               'nfsp_entropy(0.1)_pong_2p_fic',
               'nfsp_entropy(0.3)_pong_2p_fic',

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


base_path = '../data/compare_baseline_epoch10_fpo_clip_entropy(0.1)_pong_2p/match_payoff_'
fig = plt.figure(1, figsize=(7, 5.8))
plt.rc('font', size=14)
for other_name in other_names[0:6]:
    data = np.load(base_path + other_name + '.npy')
    mean = np.mean(data, axis=(1,2))
    std = np.std(data, axis=(1,2))
    plt.plot(range(0,401,10), -mean, linewidth=2)
    plt.fill_between(range(0,401,10), -mean-std, -mean+std, alpha=0.2)
plt.plot((0,400), (0,0), color='black', linestyle='--', linewidth=2)
plt.xlim(0, 400)
plt.ylim(-1, 0.4)
plt.xlabel('iterations', fontsize=18)
plt.ylabel('evaluation', fontsize=18)
plt.legend([r'SP($\epsilon$=0)',
           r'SP($\epsilon$=0.003)',
           r'SP($\epsilon$=0.01)',
           r'SP($\epsilon$=0.03)',
           r'SP($\epsilon$=0.1)',
           r'SP($\epsilon$=0.3)'], fontsize=14, loc='lower right')

plt.grid(True)


fig = plt.figure(2, figsize=(7, 5.8))
plt.rc('font', size=14)
for other_name in other_names[6:6+6]:
    data = np.load(base_path + other_name + '.npy')
    mean = np.mean(data, axis=(1,2))
    std = np.std(data, axis=(1,2))
    plt.plot(range(0,401,10), -mean, linewidth=2)
    plt.fill_between(range(0,401,10), -mean-std, -mean+std, alpha=0.2)
plt.plot((0, 400), (0, 0), color='black', linestyle='--', linewidth=2)
plt.xlim(0, 400)
plt.ylim(-1, 0.4)
plt.xlabel('iterations', fontsize=18)
plt.ylabel('evaluation', fontsize=18)
plt.legend([r'NFSP-Fic($\epsilon$=0)',
           r'NFSP-Fic($\epsilon$=0.003)',
           r'NFSP-Fic($\epsilon$=0.01)',
           r'NFSP-Fic($\epsilon$=0.03)',
           r'NFSP-Fic($\epsilon$=0.1)',
           r'NFSP-Fic($\epsilon$=0.3)'], fontsize=14, loc='upper right')

plt.grid(True)


fig = plt.figure(3, figsize=(7, 5.8))
plt.rc('font', size=14)
for other_name in other_names[12:12+6]:
    data = np.load(base_path + other_name + '.npy')
    mean = np.mean(data, axis=(1,2))
    std = np.std(data, axis=(1,2))
    plt.plot(range(0,401,10), -mean, linewidth=2)
    plt.fill_between(range(0,401,10), -mean-std, -mean+std, alpha=0.2)
plt.plot((0, 400), (0, 0), color='black', linestyle='--', linewidth=2)
plt.xlim(0, 400)
plt.ylim(-1, 0.4)
plt.xlabel('iterations', fontsize=18)
plt.ylabel('evaluation', fontsize=18)
plt.legend([r'NFSP-PPO($\epsilon$=0)',
           r'NFSP-PPO($\epsilon$=0.003)',
           r'NFSP-PPO($\epsilon$=0.01)',
           r'NFSP-PPO($\epsilon$=0.03)',
           r'NFSP-PPO($\epsilon$=0.1)',
           r'NFSP-PPO($\epsilon$=0.3)'], fontsize=14, loc='lower right')

plt.grid(True)


fig = plt.figure(4, figsize=(7, 5.8))
plt.rc('font', size=14)
for other_name in other_names[18:18+6+1]:
    data = np.load(base_path + other_name + '.npy')
    mean = np.mean(data, axis=(1,2))
    std = np.std(data, axis=(1,2))
    plt.plot(range(0,401,10), -mean, linewidth=2)
    plt.fill_between(range(0,401,10), -mean-std, -mean+std, alpha=0.2)
plt.plot((0, 400), (0, 0), color='black', linestyle='--', linewidth=2)
plt.xlim(0, 400)
plt.ylim(-1, 0.4)
plt.xlabel('iterations', fontsize=18)
plt.ylabel('evaluation', fontsize=18)
plt.legend([r'PPO($\epsilon$=0)',
           r'PPO($\epsilon$=0.003)',
           r'PPO($\epsilon$=0.01)',
           r'PPO($\epsilon$=0.03)',
           r'PPO($\epsilon$=0.1)',
           r'PPO($\epsilon$=0.3)',
            'SimpleAI'], fontsize=14, loc='upper right')

plt.grid(True)



plt.show()



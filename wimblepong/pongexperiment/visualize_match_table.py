import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import image_annotated_heatmap as iah
from scipy.linalg import schur
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['pdf.fonttype'] = 42

def visualize_match_table(match_table, c):
    print(match_table)
    fig = plt.figure()
    plt.rc('font', size=14)
    ax = sns.heatmap(match_table, linewidths=0.5, cmap='RdYlGn',
                     xticklabels=range(60, 401, 20),
                     yticklabels=range(60, 401, 20)) #, vmax=0.55, vmin=-0.55)
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-60, ha="right",
             rotation_mode="anchor")
    # im, cbar = iah.heatmap(match_table, row_labels=range(0, 401, 20),
    #                        col_labels=range(0, 401, 20),
    #                        cmap="RdYlGn", cbar_kw={"shrink": 0.90})
    # plt.show()
    plt.pause(0.01)

    U, Sigma, VT = np.linalg.svd(match_table)
    # T, Z = schur(match_table)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    ax.scatter(VT[0, :], VT[1, :], c=c, cmap='RdYlGn', edgecolors='grey', s=13**2)
    # ax.scatter(U[:,0], U[:,1], c=np.mean(match_table[:-1,:-1], axis=1), cmap='RdYlGn')
    # ax.scatter(Z[:, 0], Z[:, 1], c=c, cmap='RdYlGn')
    plt.pause(0.01)


# match_table = np.load('../data/fpo_clip_entropy(0.03)_pong_2p/fpo_clip_entropy(0.03)_pong_2p_s300_backup/match_payoff_backup.npy')
# match_table = match_table[:-1, :-1]
# c = np.mean(match_table, axis=1)
# visualize_match_table(match_table, c)
#
# match_table = np.load('../data/fpo_clip_entropy(0.03)_replay(100)_pong_2p/fpo_clip_entropy(0.03)_replay(100)_pong_2p_s300_backup/match_payoff_backup.npy')
# match_table = match_table[:-1, :-1]
# c = np.mean(match_table, axis=1)
# visualize_match_table(match_table, c)
#
# match_table = np.load('../data/fpo_clip_entropy(0.03)_replay(10)_pong_2p/fpo_clip_entropy(0.03)_replay(10)_pong_2p_s300_backup/match_payoff_backup.npy')
# match_table = match_table[:-1, :-1]
# c = np.mean(match_table, axis=1)
# visualize_match_table(match_table, c)
#
# match_table = np.load('../data/fpo_clip_entropy(0.03)_replay(1)_pong_2p/fpo_clip_entropy(0.03)_replay(1)_pong_2p_s300_backup/match_payoff_backup.npy')
# match_table = match_table[:-1, :-1]
# c = np.mean(match_table, axis=1)
# visualize_match_table(match_table, c)

# match_table = np.load('../data/fpo_clip_entropy(0.1)_replay(1)_pong_2p/fpo_clip_entropy(0.1)_replay(1)_pong_2p_s100_backup/match_payoff_backup.npy')
# match_table = match_table[3:-1, 3:-1]
# c = np.mean(match_table, axis=1)
# visualize_match_table(match_table, c)
#
# selected
match_table = np.load('../data/fpo_clip_entropy(0.1)_replay(1)_pong_2p/fpo_clip_entropy(0.1)_replay(1)_pong_2p_s200_backup/match_payoff_backup.npy')
match_table = match_table[3:-1, 3:-1]
c = np.mean(match_table, axis=1)
visualize_match_table(match_table, c)
#
# match_table = np.load('../data/fpo_clip_entropy(0.1)_replay(1)_pong_2p/fpo_clip_entropy(0.1)_replay(1)_pong_2p_s300_backup/match_payoff_backup.npy')
# match_table = match_table[3:-1, 3:-1]
# c = np.mean(match_table, axis=1)
# visualize_match_table(match_table, c)

# match_table = np.load('../data/fpo_clip_entropy(0.1)_replay(10)_pong_2p/fpo_clip_entropy(0.1)_replay(10)_pong_2p_s100_backup/match_payoff_backup.npy')
# match_table = match_table[3:-1, 3:-1]
# c = np.mean(match_table, axis=1)
# visualize_match_table(match_table, c)
#
# match_table = np.load('../data/fpo_clip_entropy(0.1)_replay(10)_pong_2p/fpo_clip_entropy(0.1)_replay(10)_pong_2p_s200_backup/match_payoff_backup.npy')
# match_table = match_table[3:-1, 3:-1]
# c = np.mean(match_table, axis=1)
# visualize_match_table(match_table, c)
#
# selected
match_table = np.load('../data/fpo_clip_entropy(0.1)_replay(10)_pong_2p/fpo_clip_entropy(0.1)_replay(10)_pong_2p_s300_backup/match_payoff_backup.npy')
match_table = match_table[3:-1, 3:-1]
c = np.mean(match_table, axis=1)
visualize_match_table(match_table, c)

# selected
match_table = np.load('../data/fpo_clip_entropy(0.1)_replay(100)_pong_2p/fpo_clip_entropy(0.1)_replay(100)_pong_2p_s100_backup/match_payoff_backup.npy')
match_table = match_table[3:-1, 3:-1]
c = np.mean(match_table, axis=1)
visualize_match_table(match_table, c)
#
# match_table = np.load('../data/fpo_clip_entropy(0.1)_replay(100)_pong_2p/fpo_clip_entropy(0.1)_replay(100)_pong_2p_s200_backup/match_payoff_backup.npy')
# match_table = match_table[3:-1, 3:-1]
# c = np.mean(match_table, axis=1)
# visualize_match_table(match_table, c)
#
# match_table = np.load('../data/fpo_clip_entropy(0.1)_replay(100)_pong_2p/fpo_clip_entropy(0.1)_replay(100)_pong_2p_s300_backup/match_payoff_backup.npy')
# match_table = match_table[3:-1, 3:-1]
# c = np.mean(match_table, axis=1)
# visualize_match_table(match_table, c)

# match_table = np.load('../data/fpo_clip_entropy(0.1)_pong_2p/fpo_clip_entropy(0.1)_pong_2p_s100_backup/match_payoff_backup.npy')
# match_table = match_table[3:-1, 3:-1]
# c = np.mean(match_table, axis=1)
# visualize_match_table(match_table, c)
#
# selected
match_table = np.load('../data/fpo_clip_entropy(0.1)_pong_2p/fpo_clip_entropy(0.1)_pong_2p_s200_backup/match_payoff_backup.npy')
match_table = match_table[3:-1, 3:-1]
c = np.mean(match_table, axis=1)
visualize_match_table(match_table, c)
#
# match_table = np.load('../data/fpo_clip_entropy(0.1)_pong_2p/fpo_clip_entropy(0.1)_pong_2p_s300_backup/match_payoff_backup.npy')
# match_table = match_table[3:-1, 3:-1]
# c = np.mean(match_table, axis=1)
# visualize_match_table(match_table, c)

plt.show()

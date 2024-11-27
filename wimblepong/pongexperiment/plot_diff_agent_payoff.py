"""
plot the value and policy for a given state, from dp results and m2qn results
heatmap inspired by https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html#some-more-complex-heatmap-examples
storyboard inspired by https://matplotlib.org/examples/pylab_examples/scatter_hist.html
colorbar inspired by https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.colorbar.html
horizontal bar inspired by https://blog.csdn.net/hohaizx/article/details/79101322

"""


import matplotlib.pyplot as plt
import image_annotated_heatmap as iah
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# plt.rcParams['pdf.fonttype'] = 14
# plt.rcParams['font.family'] = 'Arial'


data = np.load('../data/compare_diff_agents/match_payoff_backup.npy')
indices = [0] + [i for i in range(7,26)]
payoff_table = np.mean(data[indices][:, indices], axis=(2,3))
print(payoff_table)

with PdfPages('test.pdf') as pdf:
    # start with a rectangular Figure
    plt.rc('font', size=6)
    plt.rc('xtick', labelsize=9)
    plt.rc('ytick', labelsize=9)
    fig = plt.figure(1, figsize=(7, 5.8))
    im, cbar = iah.heatmap(payoff_table, row_labels=[r'EPO($\epsilon$=0.1)',

                                                     r'NFSP-BR($\epsilon$=0)',
                                                     r'NFSP-Fic($\epsilon$=0)',
                                                     r'NFSP-BR($\epsilon$=0.003)',
                                                     r'NFSP-Fic($\epsilon$=0.003)',
                                                     r'NFSP-BR($\epsilon$=0.01)',
                                                     r'NFSP-Fic($\epsilon$=0.01)',
                                                     r'NFSP-BR($\epsilon$=0.03)',
                                                     r'NFSP-Fic($\epsilon$=0.03)',
                                                     r'NFSP-BR($\epsilon$=0.1)',
                                                     r'NFSP-Fic($\epsilon$=0.1)',
                                                     r'NFSP-BR($\epsilon$=0.3)',
                                                     r'NFSP-Fic($\epsilon$=0.3)',

                                                     r'PPO($\epsilon$=0)',
                                                     r'PPO($\epsilon$=0.003)',
                                                     r'PPO($\epsilon$=0.01)',
                                                     r'PPO($\epsilon$=0.03)',
                                                     r'PPO($\epsilon$=0.1)',
                                                     r'PPO($\epsilon$=0.3)',

                                                     'SimpleAI'],
                           col_labels=[r'EPO($\epsilon$=0.1)',

                                                     r'NFSP-BR($\epsilon$=0)',
                                                     r'NFSP-Fic($\epsilon$=0)',
                                                     r'NFSP-BR($\epsilon$=0.003)',
                                                     r'NFSP-Fic($\epsilon$=0.003)',
                                                     r'NFSP-BR($\epsilon$=0.01)',
                                                     r'NFSP-Fic($\epsilon$=0.01)',
                                                     r'NFSP-BR($\epsilon$=0.03)',
                                                     r'NFSP-Fic($\epsilon$=0.03)',
                                                     r'NFSP-BR($\epsilon$=0.1)',
                                                     r'NFSP-Fic($\epsilon$=0.1)',
                                                     r'NFSP-BR($\epsilon$=0.3)',
                                                     r'NFSP-Fic($\epsilon$=0.3)',

                                                     r'PPO($\epsilon$=0)',
                                                     r'PPO($\epsilon$=0.003)',
                                                     r'PPO($\epsilon$=0.01)',
                                                     r'PPO($\epsilon$=0.03)',
                                                     r'PPO($\epsilon$=0.1)',
                                                     r'PPO($\epsilon$=0.3)',

                                                     'SimpleAI'],
                           cmap="RdYlGn", cbar_kw={"shrink":0.90})
    texts = iah.annotate_heatmap(im, valfmt="{x:.2f}", textcolors=["black", "black"])

    fig.tight_layout()
    plt.show()
    fig.savefig("plot_diff_agent_payoff.pdf")

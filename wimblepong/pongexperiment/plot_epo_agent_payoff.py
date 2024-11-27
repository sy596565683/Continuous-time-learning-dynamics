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
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['pdf.fonttype'] = 42

data = np.load('../data/compare_fpo_agents/match_payoff_backup.npy')
indices = [0,2,3,4,5,6]
payoff_table = np.mean(data[indices][:, indices], axis=(2,3))
print(payoff_table)

with PdfPages('test.pdf') as pdf:
    # start with a rectangular Figure
    plt.rc('font', size=16)
    fig = plt.figure(1, figsize=(7, 5.8))
    im, cbar = iah.heatmap(payoff_table, row_labels=[r'EPO($\epsilon$=0)',
                                                     r'EPO($\epsilon$=0.003)',
                                                     r'EPO($\epsilon$=0.01)',
                                                     r'EPO($\epsilon$=0.03)',
                                                     r'EPO($\epsilon$=0.1)',
                                                     r'EPO($\epsilon$=0.3)'],
                           col_labels=[r'EPO($\epsilon$=0)',
                                       r'EPO($\epsilon$=0.003)',
                                       r'EPO($\epsilon$=0.01)',
                                       r'EPO($\epsilon$=0.03)',
                                       r'EPO($\epsilon$=0.1)',
                                       r'EPO($\epsilon$=0.3)'],
                           cmap="RdYlGn", cbar_kw={"shrink":0.885})
    texts = iah.annotate_heatmap(im, valfmt="{x:.2f}", textcolors=["black", "black"])

    fig.tight_layout()
    plt.show()
    fig.savefig("plot_epo_agent_payoff.pdf")

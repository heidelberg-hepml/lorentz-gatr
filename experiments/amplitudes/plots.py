import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}'

FONTSIZE=14
FONTSIZE_LEGEND=13
FONTSIZE_TICK=12

colors = ["black","#0343DE","#A52A2A", "darkorange"]

def plot_histograms(file, data, labels, bins=60, xlabel=None,
                   title=None, logx=False, logy=False, xrange=None,
                   ratio_range=[.85, 1.15], ratio_ticks=[.9, 1., 1.1]):
    hists = []
    for dat in data:
        hist, bins = np.histogram(dat, bins=bins, range=xrange)
        hists.append(hist)
    integrals = [np.sum((bins[1:] - bins[:-1]) * hist) for hist in hists]
    scales = [1/integral if integral != 0. else 1. for integral in integrals]
    dup_last = lambda a: np.append(a, a[-1])

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6,4),
                            gridspec_kw={"height_ratios": [3,1], "hspace": 0.})
    for i, hist, scale, label, color in zip(range(len(hists)), hists, scales, labels, colors):
        axs[0].step(bins, dup_last(hist)*scale, label=label, color=color,
                    linewidth=1.0, where="post")
        if i==0:
            axs[0].fill_between(bins, dup_last(hist)*scale, 0.*dup_last(hist), facecolor=color,
                                alpha=.1, step="post")
            continue

        ratio = np.divide(hist * scale, hists[0] * scales[0], where=hists[0] * scales[0]!=0) # sets denominator=0 terms to 0
        axs[1].step(bins, dup_last(ratio), linewidth=1.0, where="post", color=color)

    if logx:
        axs[0].set_xscale("log")

    axs[0].legend(loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND)
    axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)
    axs[1].set_xlabel(xlabel, fontsize = FONTSIZE)

    _, ymax=axs[0].get_ylim()
    axs[0].set_ylim(0., ymax)
    axs[0].tick_params(axis="both", labelsize=FONTSIZE_TICK)
    axs[1].tick_params(axis="both", labelsize=FONTSIZE_TICK)
    axs[0].text(.04, .95, s=title, horizontalalignment="left", verticalalignment="top",
                transform=axs[0].transAxes, fontsize=FONTSIZE)

    axs[1].set_yticks(ratio_ticks)
    axs[1].set_ylim(ratio_range)
    axs[1].axhline(y=ratio_ticks[0], c="black", ls="dotted", lw=0.5)
    axs[1].axhline(y=ratio_ticks[1], c="black", ls="--", lw=0.7)
    axs[1].axhline(y=ratio_ticks[2], c="black", ls="dotted", lw=0.5)

    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()
    
def plot_single_histogram(file, data, bins=60, xlabel=None,
                   title=None, logx=False, logy=False, xrange=None):
    hist, bins = np.histogram(data, bins=bins, range=xrange)
    scale = 1/np.sum((bins[1:] - bins[:-1]) * hist)
    dup_last = lambda a: np.append(a, a[-1])

    fig, axs = plt.subplots(figsize=(6,4))
    axs.step(bins, dup_last(hist)*scale, colors[2], where="post")
    axs.fill_between(bins, dup_last(hist)*scale, 0.*dup_last(hist)*scale, facecolor=colors[2],
                                alpha=.1, step="post")

    if logx:
        axs[0].set_xscale("log")

    axs.set_xlabel(xlabel, fontsize = FONTSIZE)

    _, ymax=axs.get_ylim()
    axs.set_ylim(0., ymax)
    axs.set_xlim(xrange)
    axs.tick_params(axis="both", labelsize=FONTSIZE_TICK)
    axs.text(.04, .95, s=title, horizontalalignment="left", verticalalignment="top",
                transform=axs.transAxes, fontsize=FONTSIZE)

    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()

def plot_loss(file, losses, lr, labels=None):
    labels = [None for _ in range(len(losses))] if labels is None else labels
    iterations = range(1, len(losses[0])+1)
    fig, ax = plt.subplots()
    for loss, label in zip(losses, labels):
        ax.plot(iterations, loss, label=label)

    axright=ax.twinx()
    axright.plot(iterations, lr, label="learning rate", color="crimson")
    ax.set_xlabel("Number of iterations", fontsize=FONTSIZE)
    ax.set_ylabel("Loss", fontsize=FONTSIZE)
    axright.set_ylabel("Learning rate", fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper right")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()

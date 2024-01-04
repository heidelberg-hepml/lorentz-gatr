import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath} \usepackage{siunitx}'

FONTSIZE=14
FONTSIZE_LEGEND=13
FONTSIZE_TICK=12

colors = ["black","#0343DE","#A52A2A", "darkorange"]

def plot_roc(out, fpr, tpr, auc):
    color = colors[2]
    rnd = np.linspace(0, 1, 100)
    
    # usual roc
    fig, ax = plt.subplots(figsize=(5,4))
    ax.set_xlabel(r"$\epsilon_B$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$\epsilon_S$", fontsize=FONTSIZE)
    ax.plot(rnd, rnd, "k--")
    ax.plot(fpr, tpr, color=color)
    ax.text(.05, .95, s=f"AUC = {auc:.4f}", horizontalalignment="left", verticalalignment="top",
                transform=ax.transAxes, fontsize=FONTSIZE)
    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close()

    # physicists roc
    fig, ax = plt.subplots(figsize=(5,4))
    ax.set_xlabel(r"$\epsilon_S$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$1 / \epsilon_B$", fontsize=FONTSIZE)
    ax.set_yscale("log")
    ax.plot(rnd, 1/rnd, "k--")
    ax.plot(tpr, 1/fpr, color=color)
    ax.text(.05, .95, s=f"AUC = {auc:.4f}", horizontalalignment="left", verticalalignment="top",
                transform=ax.transAxes, fontsize=FONTSIZE)
    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close()

    # sic
    fig, ax = plt.subplots(figsize=(5,4))
    ax.set_xlabel(r"$\epsilon_S$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$\epsilon_S / \sqrt{\epsilon_B}$", fontsize=FONTSIZE)
    ax.plot(rnd, rnd**.5, "k--")
    ax.plot(tpr, tpr/fpr**.5, color=color)
    ax.text(.05, .95, s=f"AUC = {auc:.4f}", horizontalalignment="left", verticalalignment="top",
                transform=ax.transAxes, fontsize=FONTSIZE)
    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close()

    

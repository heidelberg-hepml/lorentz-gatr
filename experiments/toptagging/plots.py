import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from experiments.base_plots import plot_loss

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath} \usepackage{siunitx}'

FONTSIZE=14
FONTSIZE_LEGEND=13
FONTSIZE_TICK=12

colors = ["black","#0343DE","#A52A2A", "darkorange"]

def plot_mixer(cfg, plot_path, title, plot_dict):
    
    if cfg.plotting.loss and cfg.train:
        file = f"{plot_path}/loss.pdf"
        plot_loss(file, [plot_dict["train_loss"], plot_dict["val_loss"]], plot_dict["train_lr"],
                      labels=["train loss", "val loss"], logy=True)

    if cfg.plotting.roc:
        file = f"{plot_path}/roc.pdf"
        with PdfPages(file) as out:
            plot_roc(out, plot_dict["results_test"]["fpr"],
                     plot_dict["results_test"]["tpr"],
                     plot_dict["results_test"]["auc"],
                     title=title)

def plot_roc(out, fpr, tpr, auc, title=None):
    color = colors[2]
    rnd = np.linspace(1e-3, 1, 100)
    
    # usual roc
    fig, ax = plt.subplots(figsize=(5,4))
    ax.set_xlabel(r"$\epsilon_B$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$\epsilon_S$", fontsize=FONTSIZE)
    ax.plot(rnd, rnd, "k--")
    ax.plot(fpr, tpr, color=color)
    ax.text(.95, .05, s=f"AUC = {auc:.4f}", horizontalalignment="right", verticalalignment="bottom",
                transform=ax.transAxes, fontsize=FONTSIZE)
    ax.text(.05, .95, s=title, horizontalalignment="left", verticalalignment="top",
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
    ax.text(.05, .05, s=f"AUC = {auc:.4f}", horizontalalignment="left", verticalalignment="bottom",
                transform=ax.transAxes, fontsize=FONTSIZE)
    ax.text(.95, .95, s=title, horizontalalignment="right", verticalalignment="top",
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
    ax.text(.95, .95, s=title, horizontalalignment="right", verticalalignment="top",
                transform=ax.transAxes, fontsize=FONTSIZE)
    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close()

    

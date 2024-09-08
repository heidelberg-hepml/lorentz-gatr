import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from experiments.base_plots import plot_loss, plot_metric

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams[
    "text.latex.preamble"
] = r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath} \usepackage{siunitx}"

FONTSIZE = 14
FONTSIZE_LEGEND = 13
FONTSIZE_TICK = 12

colors = ["black", "#0343DE", "#A52A2A", "darkorange"]


def plot_mixer(cfg, plot_path, title, plot_dict):
    if cfg.plotting.loss and cfg.train:
        file = f"{plot_path}/loss.pdf"
        with PdfPages(file) as out:
            plot_loss(
                out,
                [plot_dict["train_loss"], plot_dict["val_loss"]],
                plot_dict["train_lr"],
                labels=["train loss", "val loss"],
                logy=True,
            )

    if cfg.plotting.score and cfg.evaluate:
        file = f"{plot_path}/score.pdf"
        with PdfPages(file) as out:
            plot_score(
                out,
                plot_dict["results_test"]["labels_true"],
                plot_dict["results_test"]["labels_predict"],
                title=title,
            )
            plot_score(
                out,
                plot_dict["results_test"]["labels_true"],
                plot_dict["results_test"]["labels_predict"],
                title=title,
                logy=True,
            )

    if cfg.plotting.roc and cfg.evaluate:
        file = f"{plot_path}/roc.pdf"
        with PdfPages(file) as out:
            plot_roc(
                out,
                plot_dict["results_test"]["fpr"],
                plot_dict["results_test"]["tpr"],
                plot_dict["results_test"]["auc"],
                title=title,
            )


def plot_roc(out, fpr, tpr, auc, title=None):
    color = colors[2]
    rnd = np.linspace(1e-3, 1, 100)

    # usual roc
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlabel(r"$\epsilon_B$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$\epsilon_S$", fontsize=FONTSIZE)
    ax.plot(rnd, rnd, "k--")
    ax.plot(fpr, tpr, color=color)
    ax.text(
        0.95,
        0.05,
        s=f"AUC = {auc:.4f}",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )
    ax.text(
        0.05,
        0.95,
        s=title,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )
    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close()

    # physicists roc
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlabel(r"$\epsilon_S$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$1 / \epsilon_B$", fontsize=FONTSIZE)
    ax.set_yscale("log")
    ax.plot(rnd, 1 / rnd, "k--")
    fpr_inv = 1 / fpr
    fpr_inv[~torch.isfinite(fpr_inv)] = 0.0
    ax.plot(tpr, fpr_inv, color=color)
    ax.text(
        0.05,
        0.05,
        s=f"AUC = {auc:.4f}",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )
    ax.text(
        0.95,
        0.95,
        s=title,
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )
    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close()

    # sic
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlabel(r"$\epsilon_S$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$\epsilon_S / \sqrt{\epsilon_B}$", fontsize=FONTSIZE)
    ax.plot(rnd, rnd**0.5, "k--")
    sic = tpr / fpr**0.5
    sic[~torch.isfinite(sic)] = 0.0
    ax.plot(tpr, sic, color=color)
    ax.text(
        0.95,
        0.05,
        s=f"AUC = {auc:.4f}",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )
    ax.text(
        0.05,
        0.95,
        s=title,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )
    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close()


def plot_score(
    out, labels_true, labels_predicted, title=None, xrange=[0, 1], bins=100, logy=False
):
    cols = [colors[1], colors[2]]

    fig, ax = plt.subplots(figsize=(5, 4))
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel("Normalized", fontsize=FONTSIZE)
    ax.set_xlabel("Classifier score (0=QCD, 1=top)", fontsize=FONTSIZE)
    _, bins, _ = ax.hist(
        labels_predicted[labels_true == 0],
        range=xrange,
        bins=bins,
        alpha=0.5,
        label="QCD",
        density=True,
        color=cols[0],
    )
    ax.hist(
        labels_predicted[labels_true == 1],
        bins=bins,
        label="Top",
        alpha=0.5,
        density=True,
        color=cols[1],
    )
    ax.set_xlim(xrange)
    ax.legend(frameon=False, fontsize=FONTSIZE, loc="upper left")
    ax.text(
        0.95,
        0.95,
        s=title,
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )

    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close()

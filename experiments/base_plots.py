import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath} \usepackage{siunitx}'

FONTSIZE=14
FONTSIZE_LEGEND=13
FONTSIZE_TICK=12

def plot_loss(file, losses, lr, labels=None, logy=True):
    labels = [None for _ in range(len(losses))] if labels is None else labels
    iterations = range(1, len(losses[0])+1)
    fig, ax = plt.subplots()
    for i, loss, label in zip(range(len(losses)), losses, labels):
        if len(loss) == len(iterations):
            its = iterations
        else:
            frac = len(losses[0]) / len(loss)
            its = np.arange(1, len(loss)+1) * frac
        ax.plot(its, loss, label=label)

    if logy:
        ax.set_yscale("log")
    axright=ax.twinx()
    axright.plot(iterations, lr, label="learning rate", color="crimson")
    ax.set_xlabel("Number of iterations", fontsize=FONTSIZE)
    ax.set_ylabel("Loss", fontsize=FONTSIZE)
    axright.set_ylabel("Learning rate", fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper right")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()

import numpy as np
import matplotlib.pyplot as plt

# ignore warnings here
import warnings

warnings.filterwarnings("ignore")

# load fonts
import matplotlib.font_manager as font_manager

font_dir = ["src/utils/bitstream-charter-ttf/Charter/"]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)
font_manager.findSystemFonts(fontpaths=None, fontext="ttf")

# setup matplotlib
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams[
    "text.latex.preamble"
] = r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}"

# fontsize
FONTSIZE = 14
FONTSIZE_LEGEND = 13
TICKLABELSIZE = 10


def plot_histogram(
    file,
    train,
    test,
    model,
    title,
    xlabel,
    xrange,
    model_label,
    logy=False,
    n_bins=60,
    error_range=[0.85, 1.15],
    error_ticks=[0.9, 1.0, 1.1],
):
    """
    Plotting code used for all 1d distributions

    Parameters:
    file: str
    train: np.ndarray of shape (nevents)
    test: np.ndarray of shape (nevents)
    model: np.ndarray of shape (nevents)
    title: str
    xlabel: str
    xrange: tuple with 2 floats
    model_label: str
    logy: bool
    n_bins: int
    error_range: tuple with 2 floats
    error_ticks: tuple with 3 floats
    """
    # construct labels and colors
    labels = ["Train", "Test", model_label]
    colors = ["black", "#0343DE", "#A52A2A"]

    # construct histograms
    y_trn, bins = np.histogram(train, bins=n_bins, range=xrange)
    y_tst, _ = np.histogram(test, bins=bins)
    y_mod, _ = np.histogram(model, bins=bins)
    hists = [y_trn, y_tst, y_mod]
    hist_errors = [np.sqrt(y_trn), np.sqrt(y_tst), np.sqrt(y_mod)]

    integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
    scales = [1 / integral if integral != 0.0 else 1.0 for integral in integrals]

    dup_last = lambda a: np.append(a, a[-1])

    fig, axs = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(6, 4),
        gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.00},
    )

    for i, y, y_err, scale, label, color in zip(
        range(len(hists)), hists, hist_errors, scales, labels, colors
    ):

        axs[0].step(
            bins,
            dup_last(y) * scale,
            label=label,
            color=color,
            linewidth=1.0,
            where="post",
        )
        axs[0].step(
            bins,
            dup_last(y + y_err) * scale,
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[0].step(
            bins,
            dup_last(y - y_err) * scale,
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[0].fill_between(
            bins,
            dup_last(y - y_err) * scale,
            dup_last(y + y_err) * scale,
            facecolor=color,
            alpha=0.3,
            step="post",
        )

        if label == "Train":
            axs[0].fill_between(
                bins,
                dup_last(y) * scale,
                0.0 * dup_last(y),
                facecolor=color,
                alpha=0.1,
                step="post",
            )
            continue

        ratio = (y * scale) / (hists[0] * scales[0])
        ratio_err = np.sqrt((y_err / y) ** 2 + (hist_errors[0] / hists[0]) ** 2)
        ratio_isnan = np.isnan(ratio)
        ratio[ratio_isnan] = 1.0
        ratio_err[ratio_isnan] = 0.0

        axs[1].step(bins, dup_last(ratio), linewidth=1.0, where="post", color=color)
        axs[1].step(
            bins,
            dup_last(ratio + ratio_err),
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[1].step(
            bins,
            dup_last(ratio - ratio_err),
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[1].fill_between(
            bins,
            dup_last(ratio - ratio_err),
            dup_last(ratio + ratio_err),
            facecolor=color,
            alpha=0.3,
            step="post",
        )

        delta = np.fabs(ratio - 1) * 100
        delta_err = ratio_err * 100

        markers, caps, bars = axs[2].errorbar(
            (bins[:-1] + bins[1:]) / 2,
            delta,
            yerr=delta_err,
            ecolor=color,
            color=color,
            elinewidth=0.5,
            linewidth=0,
            fmt=".",
            capsize=2,
        )
        [cap.set_alpha(0.5) for cap in caps]
        [bar.set_alpha(0.5) for bar in bars]

    axs[0].legend(loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND)
    axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)

    if logy:
        axs[0].set_yscale("log")

    _, ymax = axs[0].get_ylim()
    axs[0].set_ylim(0.0, ymax)
    axs[0].set_xlim(xrange)
    axs[0].tick_params(axis="both", labelsize=TICKLABELSIZE)
    plt.xlabel(
        r"${%s}$" % xlabel,
        fontsize=FONTSIZE,
    )
    axs[0].text(
        0.02,
        0.95,
        s=title,
        horizontalalignment="left",
        verticalalignment="top",
        transform=axs[0].transAxes,
        fontsize=FONTSIZE,
    )

    axs[1].set_ylabel(r"$\frac{\mathrm{JetGPT}}{\mathrm{Test}}$", fontsize=FONTSIZE)
    axs[1].set_yticks(error_ticks)
    axs[1].set_ylim(error_range)
    axs[1].axhline(y=error_ticks[0], c="black", ls="dotted", lw=0.5)
    axs[1].axhline(y=error_ticks[1], c="black", ls="--", lw=0.7)
    axs[1].axhline(y=error_ticks[2], c="black", ls="dotted", lw=0.5)

    axs[2].set_ylim((0.05, 20))
    axs[2].set_yscale("log")
    axs[2].set_yticks([0.1, 1.0, 10.0])
    axs[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
    axs[2].set_yticks(
        [
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
        ],
        minor=True,
    )

    axs[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
    axs[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
    axs[2].set_ylabel(r"$\delta [\%]$", fontsize=FONTSIZE)

    axs[1].tick_params(axis="both", labelsize=TICKLABELSIZE)
    axs[2].tick_params(axis="both", labelsize=TICKLABELSIZE)

    plt.savefig(file, bbox_inches="tight", format="pdf")
    plt.close()


def plot_histogram_2d(
    file,
    test,
    model,
    title,
    xlabel,
    ylabel,
    xrange,
    yrange,
    model_label,
    n_bins=100,
):
    data = [test, model]
    weights = [None, None]
    subtitles = ["Test", model_label]

    fig, axs = plt.subplots(1, len(data), figsize=(4 * len(data), 4))
    for ax, dat, weight, subtitle in zip(axs, data, weights, subtitles):
        ax.set_title(subtitle)
        ax.hist2d(
            dat[:, 0],
            dat[:, 1],
            bins=n_bins,
            range=[xrange, yrange],
            rasterized=True,
            weights=weight,
        )
        ax.set_xlabel(r"${%s}$" % xlabel)
        ax.set_ylabel(r"${%s}$" % ylabel)
    fig.suptitle(title)
    plt.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()

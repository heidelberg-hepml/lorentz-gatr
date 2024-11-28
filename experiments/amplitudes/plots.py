import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from experiments.base_plots import plot_loss

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
        plot_loss(
            file,
            [plot_dict["train_loss"], plot_dict["val_loss"]],
            plot_dict["train_lr"],
            labels=["train loss", "val loss"],
            logy=True,
        )

    if cfg.plotting.histograms and cfg.evaluate:
        out = f"{plot_path}/histograms.pdf"
        with PdfPages(out) as file:
            labels = ["Test", "Train", "Prediction"]

            for idataset, dataset in enumerate(cfg.data.dataset):
                data = [
                    np.log(plot_dict["results_test"][dataset]["raw"]["truth"]),
                    np.log(plot_dict["results_train"][dataset]["raw"]["truth"]),
                    np.log(plot_dict["results_test"][dataset]["raw"]["prediction"]),
                ]
                plot_histograms(
                    file,
                    data,
                    labels,
                    title=title[idataset],
                    xlabel=r"$\log A$",
                    logx=False,
                )

    if cfg.plotting.delta and cfg.evaluate:
        out = f"{plot_path}/delta.pdf"
        with PdfPages(out) as file:
            for idataset, dataset in enumerate(cfg.data.dataset):
                delta_test = (
                    plot_dict["results_test"][dataset]["raw"]["prediction"]
                    - plot_dict["results_test"][dataset]["raw"]["truth"]
                ) / plot_dict["results_test"][dataset]["raw"]["truth"]
                delta_train = (
                    plot_dict["results_train"][dataset]["raw"]["prediction"]
                    - plot_dict["results_train"][dataset]["raw"]["truth"]
                ) / plot_dict["results_train"][dataset]["raw"]["truth"]

                # determine 1% largest amplitudes
                scale = plot_dict["results_test"][dataset]["raw"]["truth"]
                largest_idx = round(0.01 * len(scale))
                sort_idx = np.argsort(scale)
                largest_min = scale[sort_idx][-largest_idx - 1]
                largest_mask = scale > largest_min

                xranges = [(-10.0, 10.0), (-30.0, 30.0), (-100.0, 100.0)]  # in %
                binss = [100, 50, 50]
                for xrange, bins in zip(xranges, binss):
                    plot_delta_histogram(
                        file,
                        [delta_test * 100, delta_train * 100],
                        labels=["Test", "Train"],
                        title=title[idataset],
                        xlabel=r"$\Delta = \frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}$ [\%]",
                        xrange=xrange,
                        bins=bins,
                        logy=False,
                    )
                    plot_delta_histogram(
                        file,
                        [delta_test * 100, delta_test[largest_mask] * 100],
                        labels=["Test", "Largest 1\%"],
                        title=title[idataset],
                        xlabel=r"$\Delta = \frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}$ [\%]",
                        xrange=xrange,
                        bins=bins,
                        logy=False,
                    )
                    plot_delta_histogram(
                        file,
                        [delta_test * 100, delta_test[largest_mask] * 100],
                        labels=["Test", "Largest 1\%"],
                        title=title[idataset],
                        xlabel=r"$\Delta = \frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}$ [\%]",
                        xrange=xrange,
                        bins=bins,
                        logy=True,
                    )

    if cfg.plotting.delta_prepd and cfg.evaluate:
        out = f"{plot_path}/delta_prepd.pdf"
        with PdfPages(out) as file:
            for idataset, dataset in enumerate(cfg.data.dataset):
                delta_test = (
                    plot_dict["results_test"][dataset]["preprocessed"]["prediction"]
                    - plot_dict["results_test"][dataset]["preprocessed"]["truth"]
                ) / plot_dict["results_test"][dataset]["preprocessed"]["truth"]
                delta_train = (
                    plot_dict["results_train"][dataset]["preprocessed"]["prediction"]
                    - plot_dict["results_train"][dataset]["preprocessed"]["truth"]
                ) / plot_dict["results_train"][dataset]["preprocessed"]["truth"]

                # determine 1% largest amplitudes
                scale = plot_dict["results_test"][dataset]["preprocessed"]["truth"]
                largest_idx = round(0.01 * len(scale))
                sort_idx = np.argsort(scale)
                largest_min = scale[sort_idx][-largest_idx - 1]
                largest_mask = scale > largest_min

                xranges = [(-10.0, 10.0), (-30.0, 30.0), (-100.0, 100.0)]  # in %
                binss = [100, 50, 50]
                for xrange, bins in zip(xranges, binss):
                    plot_delta_histogram(
                        file,
                        [delta_test * 100, delta_train * 100],
                        labels=["Test", "Train"],
                        title=title[idataset],
                        xlabel=r"$\tilde\Delta = \frac{\tilde A_\mathrm{pred} - \tilde A_\mathrm{true}}{\tilde A_\mathrm{true}}$ [\%]",
                        xrange=xrange,
                        bins=bins,
                        logy=False,
                    )
                    plot_delta_histogram(
                        file,
                        [delta_test * 100, delta_test[largest_mask] * 100],
                        labels=["Test", "Largest 1\%"],
                        title=title[idataset],
                        xlabel=r"$\tilde\Delta = \frac{\tilde A_\mathrm{pred} - \tilde A_\mathrm{true}}{\tilde A_\mathrm{true}}$ [\%]",
                        xrange=xrange,
                        bins=bins,
                        logy=False,
                    )
                    plot_delta_histogram(
                        file,
                        [delta_test * 100, delta_test[largest_mask] * 100],
                        labels=["Test", "Largest 1\%"],
                        title=title[idataset],
                        xlabel=r"$\tilde\Delta = \frac{\tilde A_\mathrm{pred} - \tilde A_\mathrm{true}}{\tilde A_\mathrm{true}}$ [\%]",
                        xrange=xrange,
                        bins=bins,
                        logy=True,
                    )


def plot_histograms(
    file,
    data,
    labels,
    bins=60,
    xlabel=None,
    title=None,
    logx=False,
    logy=False,
    xrange=None,
    ratio_range=[0.85, 1.15],
    ratio_ticks=[0.9, 1.0, 1.1],
):
    hists = []
    for dat in data:
        hist, bins = np.histogram(dat, bins=bins, range=xrange)
        hists.append(hist)
    integrals = [np.sum((bins[1:] - bins[:-1]) * hist) for hist in hists]
    scales = [1 / integral if integral != 0.0 else 1.0 for integral in integrals]
    dup_last = lambda a: np.append(a, a[-1])

    fig, axs = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(6, 4),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
    )
    for i, hist, scale, label, color in zip(
        range(len(hists)), hists, scales, labels, colors
    ):
        axs[0].step(
            bins,
            dup_last(hist) * scale,
            label=label,
            color=color,
            linewidth=1.0,
            where="post",
        )
        if i == 0:
            axs[0].fill_between(
                bins,
                dup_last(hist) * scale,
                0.0 * dup_last(hist),
                facecolor=color,
                alpha=0.1,
                step="post",
            )
            continue

        ratio = np.divide(
            hist * scale, hists[0] * scales[0], where=hists[0] * scales[0] != 0
        )  # sets denominator=0 terms to 0
        axs[1].step(bins, dup_last(ratio), linewidth=1.0, where="post", color=color)

    if logx:
        axs[0].set_xscale("log")

    axs[0].legend(loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND)
    axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)
    axs[1].set_xlabel(xlabel, fontsize=FONTSIZE)

    _, ymax = axs[0].get_ylim()
    axs[0].set_ylim(0.0, ymax)
    axs[0].tick_params(axis="both", labelsize=FONTSIZE_TICK)
    axs[1].tick_params(axis="both", labelsize=FONTSIZE_TICK)
    axs[0].text(
        0.04,
        0.95,
        s=title,
        horizontalalignment="left",
        verticalalignment="top",
        transform=axs[0].transAxes,
        fontsize=FONTSIZE,
    )

    axs[1].set_yticks(ratio_ticks)
    axs[1].set_ylim(ratio_range)
    axs[1].axhline(y=ratio_ticks[0], c="black", ls="dotted", lw=0.5)
    axs[1].axhline(y=ratio_ticks[1], c="black", ls="--", lw=0.7)
    axs[1].axhline(y=ratio_ticks[2], c="black", ls="dotted", lw=0.5)

    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def plot_delta_histogram(
    file, datas, labels, title, xrange, bins=60, xlabel=None, logy=False
):
    assert len(datas) == 2
    dup_last = lambda a: np.append(a, a[-1])
    _, bins = np.histogram(datas[0], bins=bins - 1, range=xrange)
    hists, scales, mses = [], [], []
    for data in datas:
        mse = np.mean(data**2)

        data = np.clip(data, xrange[0], xrange[1])
        hist, _ = np.histogram(data, bins=bins, range=xrange)
        scale = 1 / np.sum((bins[1:] - bins[:-1]) * hist)
        mses.append(mse)
        hists.append(hist)
        scales.append(scale)

    fig, axs = plt.subplots(figsize=(6, 4))
    for hist, scale, mse, label, color in zip(
        hists, scales, mses, labels, colors[1:3][::-1]
    ):
        axs.step(
            bins,
            dup_last(hist) * scale,
            color,
            where="post",
            label=label + r" ($\overline{\Delta^2} = {%.2g})$" % (mse * 1e-4),
        )  # need 1e-4 to compensate for initial *100
        axs.fill_between(
            bins,
            dup_last(hist) * scale,
            0.0 * dup_last(hist) * scale,
            facecolor=color,
            alpha=0.1,
            step="post",
        )

    if logy:
        axs.set_yscale("log")
    ymin, ymax = axs.get_ylim()
    if not logy:
        ymin = 0.0
    axs.vlines(0.0, ymin, ymax, color="k", linestyle="--", lw=0.5)
    axs.set_ylim(ymin, ymax)
    axs.set_xlim(xrange)

    axs.set_xlabel(xlabel, fontsize=FONTSIZE)
    axs.tick_params(axis="both", labelsize=FONTSIZE_TICK)
    axs.legend(frameon=False, loc="upper left", fontsize=FONTSIZE * 0.7)
    axs.text(
        0.95,
        0.95,
        s=title,
        horizontalalignment="right",
        verticalalignment="top",
        transform=axs.transAxes,
        fontsize=FONTSIZE,
    )

    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()

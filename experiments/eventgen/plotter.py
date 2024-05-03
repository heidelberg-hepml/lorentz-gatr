import numpy as np
import math
from matplotlib.backends.backend_pdf import PdfPages

from experiments.eventgen.helpers import (
    delta_eta,
    delta_phi,
    delta_r,
    get_virtual_particle,
    fourmomenta_to_jetmomenta,
)
from experiments.base_plots import plot_loss
from experiments.eventgen.plots import (
    plot_histogram,
    plot_histogram_2d,
    plot_calibration,
    simple_histogram,
    plot_roc,
)


def plot_losses(exp, filename, model_label):
    with PdfPages(filename) as file:
        plot_loss(
            file,
            [exp.train_loss, exp.val_loss],
            exp.train_lr,
            labels=["train loss", "val loss"],
            logy=True,
        )

        for ijet, n_jets in enumerate(exp.cfg.data.n_jets):
            plot_loss(
                file,
                [
                    exp.train_metrics[f"{n_jets}j.mse"],
                    exp.val_metrics[f"{n_jets}j.mse"],
                ],
                lr=exp.train_lr,
                labels=[f"train mse {n_jets}j", f"val mse {n_jets}j"],
                logy=True,
            )
            for k in range(4):
                plot_loss(
                    file,
                    [
                        exp.train_metrics[f"{n_jets}j.mse_{k}"],
                        exp.val_metrics[f"{n_jets}j.mse_{k}"],
                    ],
                    lr=exp.train_lr,
                    labels=[f"train mse_{k} {n_jets}j", f"val mse_{k} {n_jets}j"],
                    logy=True,
                )


def plot_classifier(exp, filename, model_label):
    with PdfPages(filename) as file:
        for n_jets, ijet in enumerate(exp.cfg.data.n_jets):
            # classifier train and validation loss
            plot_loss(
                file,
                [exp.classifiers[ijet].tracker[key] for key in ["loss", "val_loss"]],
                lr=exp.classifiers[ijet].tracker["lr"],
                labels=[f"train mse {n_jets}j", f"val mse {n_jets}j"],
                logy=True,
            )

            # probabilities
            data = [
                exp.classifiers[ijet].results["logits"]["true"],
                exp.classifiers[ijet].results["logits"]["fake"],
            ]
            simple_histogram(
                file,
                data,
                labels=["Test", "Generator"],
                xrange=[0, 1],
                xlabel="Classifier score",
                logx=False,
                logy=False,
            )
            simple_histogram(
                file,
                data,
                labels=["Test", "Generator"],
                xrange=[0, 1],
                xlabel="Classifier score",
                logx=False,
                logy=True,
            )

            # weights
            data = [
                exp.classifiers[ijet].results["weights"]["true"],
                exp.classifiers[ijet].results["weights"]["fake"],
            ]
            simple_histogram(
                file,
                data,
                labels=["Test", "Generator"],
                xrange=[0, 5],
                xlabel="Classifier weights",
                logx=False,
                logy=False,
            )
            simple_histogram(
                file,
                data,
                labels=["Test", "Generator"],
                xrange=[1e-3, 1e2],
                xlabel="Classifier weights",
                logx=True,
                logy=True,
            )

            # roc curve
            plot_roc(
                file,
                exp.classifiers[ijet].results["tpr"],
                exp.classifiers[ijet].results["fpr"],
                exp.classifiers[ijet].results["auc"],
            )
            # calibration curve
            plot_calibration(
                file,
                exp.classifiers[ijet].results["prob_true"],
                exp.classifiers[ijet].results["prob_pred"],
            )


def plot_fourmomenta(exp, filename, model_label):
    obs_names = []
    for name in exp.obs_names_index:
        obs_names.extend(
            [
                "E_{" + name + "}",
                "p_{x," + name + "}",
                "p_{y," + name + "}",
                "p_{z," + name + "}",
            ]
        )

    with PdfPages(filename) as file:
        for ijet in range(len(exp.cfg.data.n_jets)):
            num_components = 4 * (exp.n_hard_particles + exp.cfg.data.n_jets[ijet])
            for channel in range(num_components):

                def extract(event):
                    event = event.clone()
                    event = event.reshape(event.shape[0], -1)[:, channel]
                    return event

                train = extract(exp.data_raw[ijet]["trn"])
                test = extract(exp.data_raw[ijet]["tst"])
                model = extract(exp.data_raw[ijet]["gen"])
                xlabel = obs_names[channel]
                xrange = exp.fourmomentum_ranges[channel % 4]
                logy = False
                plot_histogram(
                    file=file,
                    train=train,
                    test=test,
                    model=model,
                    title=exp.plot_titles[ijet],
                    xlabel=xlabel,
                    xrange=xrange,
                    logy=logy,
                    model_label=model_label,
                )


def plot_jetmomenta(exp, filename, model_label):
    obs_names = []
    for name in exp.obs_names_index:
        obs_names.extend(
            [
                "p_{T," + name + "}",
                "\phi_{" + name + "}",
                "\eta_{" + name + "}",
                "m_{" + name + "}",
            ]
        )
    logys = [True, False, False, False]

    with PdfPages(filename) as file:
        for ijet in range(len(exp.cfg.data.n_jets)):
            num_components = 4 * (exp.n_hard_particles + exp.cfg.data.n_jets[ijet])
            for channel in range(num_components):

                def extract(event):
                    event = event.clone()
                    event = fourmomenta_to_jetmomenta(event)
                    event = event.reshape(event.shape[0], -1)[:, channel]
                    return event

                train = extract(exp.data_raw[ijet]["trn"])
                test = extract(exp.data_raw[ijet]["tst"])
                model = extract(exp.data_raw[ijet]["gen"])
                xlabel = obs_names[channel]
                xrange = exp.jetmomentum_ranges[channel % 4]
                logy = logys[channel % 4]
                plot_histogram(
                    file=file,
                    train=train,
                    test=test,
                    model=model,
                    title=exp.plot_titles[ijet],
                    xlabel=xlabel,
                    xrange=xrange,
                    logy=logy,
                    model_label=model_label,
                )


def plot_preprocessed(exp, filename, model_label):
    with PdfPages(filename) as file:
        for ijet in range(len(exp.cfg.data.n_jets)):

            def extract(x, channel):
                x = exp.model.coordinates_sampling.fourmomenta_to_x(x)
                x = x.reshape(x.shape[0], -1)
                return x[:, channel]

            for channel in range(
                exp.data_prepd[ijet]["trn"]
                .reshape(exp.data_prepd[ijet]["trn"].shape[0], -1)
                .shape[1]
            ):
                train = extract(exp.data_prepd[ijet]["trn"], channel)
                test = extract(exp.data_prepd[ijet]["tst"], channel)
                model = extract(exp.data_prepd[ijet]["gen"], channel)
                xlabel = r"\mathrm{channel}\ " + str(channel)
                xrange = [-5, 5]
                logy = False
                plot_histogram(
                    file=file,
                    train=train,
                    test=test,
                    model=model,
                    title=exp.plot_titles[ijet],
                    xlabel=xlabel,
                    xrange=xrange,
                    logy=logy,
                    model_label=model_label,
                )


def plot_delta(exp, filename, model_label):
    with PdfPages(filename) as file:
        for ijet in range(len(exp.cfg.data.n_jets)):
            num_particles = exp.n_hard_particles + exp.cfg.data.n_jets[ijet]
            particles = np.arange(num_particles)

            for idx1 in particles:
                for idx2 in particles:
                    if idx1 >= idx2:
                        continue

                    # delta eta
                    get_delta_eta = lambda x: delta_eta(
                        fourmomenta_to_jetmomenta(x), idx1, idx2
                    )
                    xlabel = (
                        r"\Delta \eta_{%s}"
                        % f"{exp.obs_names_index[idx1]},{exp.obs_names_index[idx2]}"
                    )
                    xrange = [-6.0, 6.0]
                    train = get_delta_eta(exp.data_raw[ijet]["trn"])
                    test = get_delta_eta(exp.data_raw[ijet]["tst"])
                    model = get_delta_eta(exp.data_raw[ijet]["gen"])
                    plot_histogram(
                        file=file,
                        train=train,
                        test=test,
                        model=model,
                        title=exp.plot_titles[ijet],
                        xlabel=xlabel,
                        xrange=xrange,
                        logy=False,
                        model_label=model_label,
                    )

                    # delta phi
                    get_delta_phi = lambda x: delta_phi(
                        fourmomenta_to_jetmomenta(x), idx1, idx2
                    )
                    xlabel = (
                        r"\Delta \phi_{%s}"
                        % f"{exp.obs_names_index[idx1]},{exp.obs_names_index[idx2]}"
                    )
                    xrange = [-math.pi, math.pi]
                    train = get_delta_phi(exp.data_raw[ijet]["trn"])
                    test = get_delta_phi(exp.data_raw[ijet]["tst"])
                    model = get_delta_phi(exp.data_raw[ijet]["gen"])
                    plot_histogram(
                        file=file,
                        train=train,
                        test=test,
                        model=model,
                        title=exp.plot_titles[ijet],
                        xlabel=xlabel,
                        xrange=xrange,
                        logy=False,
                        model_label=model_label,
                    )

                    # delta R
                    get_delta_r = lambda x: delta_r(
                        fourmomenta_to_jetmomenta(x), idx1, idx2
                    )
                    xlabel = (
                        r"\Delta R_{%s}"
                        % f"{exp.obs_names_index[idx1]},{exp.obs_names_index[idx2]}"
                    )
                    xrange = [0.0, 8.0]
                    train = get_delta_r(exp.data_raw[ijet]["trn"])
                    test = get_delta_r(exp.data_raw[ijet]["tst"])
                    model = get_delta_r(exp.data_raw[ijet]["gen"])
                    plot_histogram(
                        file=file,
                        train=train,
                        test=test,
                        model=model,
                        title=exp.plot_titles[ijet],
                        xlabel=xlabel,
                        xrange=xrange,
                        logy=False,
                        model_label=model_label,
                    )


def plot_virtual(exp, filename, model_label):
    logys = [True, False, False, False]
    with PdfPages(filename) as file:
        for ijet in range(len(exp.cfg.data.n_jets)):
            for i, components in enumerate(exp.virtual_components):
                get_virtual = lambda x: get_virtual_particle(
                    fourmomenta_to_jetmomenta(x), components
                )
                train = get_virtual(exp.data_raw[ijet]["trn"])
                test = get_virtual(exp.data_raw[ijet]["tst"])
                model = get_virtual(exp.data_raw[ijet]["gen"])
                for j in range(4):
                    plot_histogram(
                        file=file,
                        train=train[:, j],
                        test=test[:, j],
                        model=model[:, j],
                        title=exp.plot_titles[ijet],
                        xlabel=exp.virtual_names[4 * i + j],
                        xrange=exp.virtual_ranges[4 * i + j],
                        logy=logys[j],
                        model_label=model_label,
                    )


def plot_deta_dphi(exp, filename, model_label):
    with PdfPages(filename) as file:
        for ijet in range(len(exp.cfg.data.n_jets)):
            num_particles = exp.n_hard_particles + exp.cfg.data.n_jets[ijet]
            particles = np.arange(num_particles)

            for idx1 in particles:
                for idx2 in particles:
                    if idx1 >= idx2:
                        continue

                    def construct(event):
                        deta = delta_eta(fourmomenta_to_jetmomenta(event), idx1, idx2)
                        dphi = delta_phi(fourmomenta_to_jetmomenta(event), idx1, idx2)
                        return np.stack([deta, dphi], axis=-1)

                    test = construct(exp.data_raw[ijet]["tst"])
                    model = construct(exp.data_raw[ijet]["gen"])
                    xlabel = (
                        r"\Delta \eta_{%s}"
                        % f"{exp.obs_names_index[idx1]},{exp.obs_names_index[idx2]}"
                    )
                    ylabel = (
                        r"\Delta \phi_{%s}"
                        % f"{exp.obs_names_index[idx1]},{exp.obs_names_index[idx2]}"
                    )
                    xrange = [-4.0, 4.0]
                    yrange = [-np.pi, np.pi]
                    plot_histogram_2d(
                        file=file,
                        test=test,
                        model=model,
                        title=exp.plot_titles[ijet],
                        xlabel=xlabel,
                        ylabel=ylabel,
                        xrange=xrange,
                        yrange=yrange,
                        model_label=model_label,
                    )

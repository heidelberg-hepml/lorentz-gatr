import numpy as np
import torch

import os, time
from omegaconf import open_dict
from hydra.utils import instantiate
from tqdm import trange, tqdm

from experiments.base_experiment import BaseExperiment
from experiments.eventgen.dataset import EventDataset, EventDataLoader
from experiments.eventgen.helpers import ensure_onshell
import experiments.eventgen.plotter as plotter
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow


class EventGenerationExperiment(BaseExperiment):
    def init_physics(self):
        self.define_process_specifics()

        # dynamically set wrapper properties
        self.modeltype = (
            "GPT" if self.cfg.model._target_.rsplit(".", 1)[-1] == "JetGPT" else "CFM"
        )
        self.modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
        n_particles = self.n_hard_particles + max(self.cfg.data.n_jets)
        n_datasets = len(self.cfg.data.n_jets)
        with open_dict(self.cfg):
            # preparation for joint training
            if self.modelname in ["GATr", "Transformer"]:
                self.cfg.model.process_token_channels = n_datasets
                self.cfg.model.type_token_channels = n_particles
            else:
                # no joint training possible
                assert len(self.cfg.data.n_jets) == 1

            # dynamically set channel dimensions
            if self.modelname == "GATr":
                self.cfg.model.net.in_s_channels = (
                    n_particles + n_datasets + self.cfg.cfm.embed_t_dim
                )
            elif self.modelname == "Transformer":
                self.cfg.model.net.in_channels = (
                    4 + n_datasets + n_particles + self.cfg.cfm.embed_t_dim
                )
            elif self.modelname == "GAP":
                self.cfg.model.net.in_mv_channels = n_particles
                self.cfg.model.net.out_mv_channels = n_particles
                self.cfg.model.net.in_s_channels = self.cfg.cfm.embed_t_dim
                self.cfg.model.net.out_s_channels = n_particles * 4
            elif self.modelname == "MLP":
                self.cfg.model.net.in_shape = 4 * n_particles + self.cfg.cfm.embed_t_dim
                self.cfg.model.net.out_shape = 4 * n_particles

            # extra treatment for lorentz-symmetry breaking inputs in equivariant models
            if self.modelname in ["GATr", "GAP"]:
                if self.cfg.model.beam_reference is not None:
                    self.cfg.model.net.in_mv_channels += (
                        2
                        if (
                            self.cfg.model.two_beams
                            and self.cfg.model.beam_reference != "xyplane"
                        )
                        else 1
                    )
                if self.cfg.model.add_time_reference:
                    self.cfg.model.net.in_mv_channels += 1

            # copy model-specific parameters
            if self.modeltype == "CFM":
                self.cfg.model.odeint = self.cfg.odeint
                self.cfg.model.cfm = self.cfg.cfm
            elif self.modeltype == "GPT":
                assert (
                    self.cfg.exp_type == "ttbar"
                ), "JetGPT only implemented for exp_type=ttbar, not exp_type={self.cfg.exp_type}"
                self.cfg.model.gpt = self.cfg.gpt
                if self.cfg.model.n_gauss is None:
                    self.cfg.model.n_gauss = self.cfg.model.net.hidden_channels // 3
                max_idx = 4 * n_particles
                self.cfg.model.net.in_channels = 1 + max_idx + n_datasets
                self.cfg.model.net.out_channels = 3 * self.cfg.model.n_gauss
            else:
                raise ValueError(f"modeltype={self.modeltype} not implemented")

    def init_data(self):
        LOGGER.info(f"Working with {self.cfg.data.n_jets} extra jets")

        # load all datasets and organize them in lists
        self.events_raw = []
        for n_jets in self.cfg.data.n_jets:
            # load data
            data_path = eval(f"self.cfg.data.data_path_{n_jets}j")
            assert os.path.exists(data_path), f"data_path {data_path} does not exist"
            data_raw = np.load(data_path)
            LOGGER.info(f"Loaded data with shape {data_raw.shape} from {data_path}")

            # bring data into correct shape
            if self.cfg.data.subsample is not None:
                assert self.cfg.data.subsample < data_raw.shape[0]
                LOGGER.info(
                    f"Reducing the size of the dataset from {data_raw.shape[0]} to {self.cfg.data.subsample}"
                )
                data_raw = data_raw[: self.cfg.data.subsample, :]
            data_raw = data_raw.reshape(data_raw.shape[0], data_raw.shape[1] // 4, 4)
            data_raw = torch.tensor(data_raw, dtype=self.dtype)

            # collect everything
            self.events_raw.append(data_raw)

        # change global units
        self.model.init_physics(
            self.units,
            self.pt_min,
            self.delta_r_min,
            self.onshell_list,
            self.onshell_mass,
            self.virtual_components,
            self.cfg.data.base_type,
            self.cfg.data.use_pt_min,
            self.cfg.data.use_delta_r_min,
        )

        # preprocessing
        self.events_prepd = []
        for ijet in range(len(self.cfg.data.n_jets)):
            # preprocess data
            self.events_raw[ijet] = ensure_onshell(
                self.events_raw[ijet],
                self.onshell_list,
                self.onshell_mass,
            )
            data_prepd = self.model.preprocess(self.events_raw[ijet])
            self.events_prepd.append(data_prepd)

        # initialize cfm (might require data)
        self.model.init_distribution()
        self.model.init_coordinates()
        fit_data = [x / self.units for x in self.events_raw]
        for coordinates in self.model.coordinates:
            coordinates.init_fit(fit_data)
        if hasattr(self.model, "distribution"):
            self.model.distribution.coordinates.init_fit(fit_data)

        plot_path = (
            os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
            if self.cfg.save and self.cfg.plot
            else None
        )
        self.model.init_anything(
            fit_data, plot_path=plot_path, device=self.device, dtype=self.dtype
        )

    def _init_dataloader(self):
        assert sum(self.cfg.data.train_test_val) <= 1

        # seperate data into train, test and validation subsets for each dataset
        self.data_raw, self.data_prepd = [], []
        for ijet, n_jets in enumerate(self.cfg.data.n_jets):
            n_data = self.events_raw[ijet].shape[0]
            split_val = int(n_data * self.cfg.data.train_test_val[::-1][0])
            split_test = int(n_data * sum(self.cfg.data.train_test_val[::-1][:2]))
            split_train = int(n_data * sum(self.cfg.data.train_test_val[::-1]))

            data_raw = {
                "val": self.events_raw[ijet][0:split_val],
                "tst": self.events_raw[ijet][split_val:split_test],
                "trn": self.events_raw[ijet][split_test:split_train],
            }
            data_prepd = {
                "val": self.events_prepd[ijet][0:split_val],
                "tst": self.events_prepd[ijet][split_val:split_test],
                "trn": self.events_prepd[ijet][split_test:split_train],
            }
            self.data_raw.append(data_raw)
            self.data_prepd.append(data_prepd)

        # create dataloaders
        self.train_loader = EventDataLoader(
            dataset=EventDataset([x["trn"] for x in self.data_prepd], dtype=self.dtype),
            batch_size=self.cfg.training.batchsize,
            shuffle=True,
        )

        self.test_loader = EventDataLoader(
            dataset=EventDataset([x["tst"] for x in self.data_prepd], dtype=self.dtype),
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

        self.val_loader = EventDataLoader(
            dataset=EventDataset([x["val"] for x in self.data_prepd], dtype=self.dtype),
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

        LOGGER.info(
            f"Constructed dataloaders with train_test_val={self.cfg.data.train_test_val}, "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

    @torch.no_grad()
    def evaluate(self):
        # EMA-evaluation not implemented
        loaders = {
            "train": self.train_loader,
            "test": self.test_loader,
            "val": self.val_loader,
        }
        if self.cfg.evaluation.sample:
            self._sample_events()
            loaders["gen"] = self.sample_loader
        else:
            LOGGER.info("Skip sampling")

        if self.cfg.evaluation.classifier:
            self.classifiers = []
            for ijet, n_jets in enumerate(self.cfg.data.n_jets):
                self.classifiers.append(self._evaluate_classifier_metric(ijet, n_jets))

        for key in self.cfg.evaluation.eval_loss:
            if key in loaders.keys():
                self._evaluate_loss_single(loaders[key], key)
        for key in self.cfg.evaluation.eval_log_prob:
            if key == "gen":
                # log_probs of generated events are not interesting
                # + they are not well-defined, because generated events might be in regions
                # that are not included in the base distribution (because of pt_min, delta_r_min)
                continue
            self._evaluate_log_prob_single(loaders[key], key)

    def _evaluate_classifier_metric(self, ijet, n_jets):
        assert self.cfg.evaluation.sample, "need samples for classifier evaluation"

        # initiate
        with open_dict(self.cfg):
            num_particles = self.n_hard_particles + n_jets
            self.cfg.classifier.net.in_shape = 4 * num_particles
            if self.cfg.classifier.cfg_preprocessing.add_delta_r:
                self.cfg.classifier.net.in_shape += (
                    num_particles * (num_particles - 1) // 2
                )
            if self.cfg.classifier.cfg_preprocessing.add_virtual:
                self.cfg.classifier.net.in_shape += 4 * len(self.virtual_components)
        classifier_factory = instantiate(self.cfg.classifier, _partial_=True)
        classifier = classifier_factory(experiment=self, device=self.device)

        data_true = self.events_raw[ijet]
        data_fake = self.data_raw[ijet]["gen"]
        LOGGER.info(
            f"Classifier generated data true/fake has shape {tuple(data_true.shape)}/{tuple(data_fake.shape)}"
        )

        # preprocessing
        cls_params = {"mean": None, "std": None}
        data_true, cls_params = classifier.preprocess(data_true, cls_params)
        data_fake = classifier.preprocess(data_fake, cls_params)[0]
        data_true = classifier.train_test_val_split(data_true)
        data_fake = classifier.train_test_val_split(data_fake)
        classifier.init_data(data_true, data_fake)

        # do things
        classifier.train()
        classifier.evaluate()

        # save weighted events
        if self.cfg.evaluation.save_samples and self.cfg.save:
            events_true = classifier.train_test_val_split(self.events_raw[ijet])["tst"]
            events_fake = classifier.train_test_val_split(self.data_raw[ijet]["gen"])[
                "tst"
            ]
            weights_true = classifier.results["weights"]["true"]
            weights_fake = classifier.results["weights"]["fake"]
            os.makedirs(os.path.join(self.cfg.run_dir, "samples"), exist_ok=True)
            filename = os.path.join(
                self.cfg.run_dir,
                "samples",
                f"samples_weighted_{self.cfg.run_idx}_{n_jets}j",
            )
            np.savez(
                filename,
                events_true=events_true,
                events_fake=events_fake,
                weights_true=weights_true,
                weights_fake=weights_fake,
            )
        return classifier

    def _evaluate_loss_single(self, loader, title):
        self.model.eval()
        losses = []
        mses = {f"{n_jets}j": [] for n_jets in self.cfg.data.n_jets}
        LOGGER.info(f"Starting to evaluate loss for model on {title} dataset")
        t0 = time.time()
        for i, data in enumerate(loader):
            loss = 0.0
            for ijet, data_single in enumerate(data):
                x0 = data_single.to(self.device)
                loss_single = self.model.batch_loss(x0, ijet)[0]
                loss += loss_single / len(self.cfg.data.n_jets)
                mses[f"{self.cfg.data.n_jets[ijet]}j"].append(loss_single.cpu().item())
            losses.append(loss.cpu().item())
        dt = time.time() - t0
        LOGGER.info(
            f"Finished evaluating loss for {title} dataset after {dt/60:.2f}min"
        )

        if self.cfg.use_mlflow:
            log_mlflow(f"eval.{title}.loss", np.mean(losses))
            for key, values in mses.items():
                log_mlflow(f"eval.{title}.{key}.mse", np.mean(values))

    def _evaluate_log_prob_single(self, loader, title):
        self.model.eval()
        self.NLLs = {f"{n_jets}j": [] for n_jets in self.cfg.data.n_jets}
        LOGGER.info(f"Starting to evaluate log_prob for model on {title} dataset")
        t0 = time.time()
        for i, data in enumerate(tqdm(loader)):
            for ijet, data_single in enumerate(data):
                x0 = data_single.to(self.device)
                NLL = -self.model.log_prob(x0, ijet).squeeze().cpu()
                self.NLLs[f"{self.cfg.data.n_jets[ijet]}j"].extend(
                    NLL.squeeze().numpy().tolist()
                )
        dt = time.time() - t0
        LOGGER.info(
            f"Finished evaluating log_prob for {title} dataset after {dt/60:.2f}min"
        )
        for key, values in self.NLLs.items():
            LOGGER.info(f"NLL_{key} = {np.mean(values)}")
            if self.cfg.use_mlflow:
                log_mlflow(f"eval.{title}.{key}.NLL", np.mean(values))

    def _sample_events(self):
        self.model.eval()

        for ijet, n_jets in enumerate(self.cfg.data.n_jets):
            sample = []
            shape = (self.cfg.evaluation.batchsize, self.n_hard_particles + n_jets, 4)
            n_batches = (
                1 + (self.cfg.evaluation.nsamples - 1) // self.cfg.evaluation.batchsize
            )
            LOGGER.info(
                f"Starting to generate {self.cfg.evaluation.nsamples} {n_jets}j events"
            )
            t0 = time.time()
            for i in trange(n_batches, desc="Sampled batches"):
                x_t = self.model.sample(
                    ijet,
                    shape,
                    self.device,
                    self.dtype,
                )
                sample.append(x_t)
            t1 = time.time()
            LOGGER.info(
                f"Finished generating {n_jets}j events after {t1-t0:.2f}s = {(t1-t0)/60:.2f}min"
            )

            samples = torch.cat(sample, dim=0)[
                : self.cfg.evaluation.nsamples, ...
            ].cpu()
            self.data_prepd[ijet]["gen"] = samples

            samples_raw = self.model.undo_preprocess(samples)
            self.data_raw[ijet]["gen"] = samples_raw

            m2 = samples_raw[..., 0] ** 2 - (samples_raw[..., 1:] ** 2).sum(dim=-1)
            LOGGER.info(
                f"Fraction of events with m2<0: {(m2<0).float().mean():.4f} (flip m2->-m2 for these events)"
            )

            if self.cfg.evaluation.save_samples and self.cfg.save:
                os.makedirs(os.path.join(self.cfg.run_dir, "samples"), exist_ok=True)
                filename = os.path.join(
                    self.cfg.run_dir,
                    "samples",
                    f"samples_{self.cfg.run_idx}_{n_jets}j.npy",
                )
                np.save(filename, samples_raw)

        self.sample_loader = torch.utils.data.DataLoader(
            dataset=EventDataset([x["gen"] for x in self.data_prepd], dtype=self.dtype),
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

    def plot(self):
        path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(os.path.join(path), exist_ok=True)
        LOGGER.info(f"Creating plots in {path}")

        self.plot_titles = [
            r"${%s}+{%s}j$" % (self.plot_title, n_jets)
            for n_jets in self.cfg.data.n_jets
        ]
        kwargs = {
            "exp": self,
            "model_label": self.modelname,
        }

        if self.cfg.train:
            filename = os.path.join(path, "loss.pdf")
            plotter.plot_losses(filename=filename, **kwargs)

        if not self.cfg.evaluate:
            return

        # set correct masses
        if self.cfg.evaluation.sample:
            for label in ["trn", "tst", "gen"]:
                for ijet in range(len(self.cfg.data.n_jets)):
                    self.data_raw[ijet][label] = ensure_onshell(
                        self.data_raw[ijet][label],
                        self.onshell_list,
                        self.onshell_mass,
                    )

        # If specified, collect weights from classifier
        if self.cfg.evaluation.classifier and self.cfg.plotting.reweighted:
            weights = {
                ijet: self.classifiers[ijet].weights_fake.squeeze()
                for ijet, n_jets in enumerate(self.cfg.data.n_jets)
            }
        else:
            weights = {ijet: None for ijet, n_jets in enumerate(self.cfg.data.n_jets)}

        # can manually create a mask
        if self.cfg.plotting.create_mask:
            mask_dict = {}
            for ijet, n_jets in enumerate(self.cfg.data.n_jets):
                # create your mask here
                assert weights[ijet] is not None
                mask_dict[ijet] = {"condition": "w<0.5", "mask": weights[ijet] < 0.5}
            weights[ijet] = None
        else:
            mask_dict = {ijet: None for ijet, n_jets in enumerate(self.cfg.data.n_jets)}

        if (
            self.cfg.plotting.log_prob
            and len(self.cfg.evaluation.eval_log_prob) > 0
            and self.cfg.evaluate
        ):
            filename = os.path.join(path, "neg_log_prob.pdf")
            plotter.plot_log_prob(filename=filename, **kwargs)

        if self.cfg.evaluation.classifier and self.cfg.evaluate:
            filename = os.path.join(path, "classifier.pdf")
            plotter.plot_classifier(filename=filename, **kwargs)

        if self.cfg.evaluation.sample:
            if self.cfg.plotting.fourmomenta:
                filename = os.path.join(path, "fourmomenta.pdf")
                plotter.plot_fourmomenta(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )

            if self.cfg.plotting.jetmomenta:
                filename = os.path.join(path, "jetmomenta.pdf")
                plotter.plot_jetmomenta(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )

            if self.cfg.plotting.preprocessed:
                filename = os.path.join(path, "preprocessed.pdf")
                plotter.plot_preprocessed(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )

            if self.cfg.plotting.virtual and len(self.virtual_components) > 0:
                filename = os.path.join(path, "virtual.pdf")
                plotter.plot_virtual(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )

            if self.cfg.plotting.delta:
                filename = os.path.join(path, "delta.pdf")
                plotter.plot_delta(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )

            if self.cfg.plotting.deta_dphi:
                filename = os.path.join(path, "deta_dphi.pdf")
                plotter.plot_deta_dphi(filename=filename, **kwargs)

    def _init_loss(self):
        # loss defined manually within the model
        pass

    def _batch_loss(self, data):

        # average over contributions from different datasets
        loss = 0.0
        mse = []
        component_mse = []
        for ijet, x0 in enumerate(data):
            x0 = x0.to(self.device)
            mse_single, component_mse_single = self.model.batch_loss(x0, ijet)
            loss += mse_single / len(self.cfg.data.n_jets)
            mse.append(mse_single.cpu().item())
            component_mse.append([x.cpu().item() for x in component_mse_single])
        assert torch.isfinite(loss).all()

        metrics = {
            f"{n_jets}j.mse": mse[ijet]
            for (ijet, n_jets) in enumerate(self.cfg.data.n_jets)
        }
        for k in range(4):
            for ijet, n_jets in enumerate(self.cfg.data.n_jets):
                metrics[f"{n_jets}j.mse_{k}"] = component_mse[ijet][k]
        return loss, metrics

    def _init_metrics(self):
        metrics = {f"{n_jets}j.mse": [] for n_jets in self.cfg.data.n_jets}
        for k in range(4):
            for n_jets in self.cfg.data.n_jets:
                metrics[f"{n_jets}j.mse_{k}"] = []
        return metrics

    def define_process_specifics(self):
        self.plot_title = None
        self.n_hard_particles = None
        self.n_jets_max = None
        self.onshell_list = None
        self.onshell_mass = None
        self.units = None
        self.pt_min = None
        self.delta_r_min = None
        self.obs_names_index = None
        self.fourmomentum_ranges = None
        self.jetmomentum_ranges = None
        self.virtual_components = None
        self.virtual_names = None
        self.virtual_ranges = None
        raise NotImplementedError

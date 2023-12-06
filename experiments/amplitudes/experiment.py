import numpy as np
import torch
import time
from datetime import datetime
import os, sys
import zipfile
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from matplotlib.backends.backend_pdf import PdfPages

from experiments.misc import get_device, save_config
from experiments.amplitudes.wrappers import AmplitudeMLPWrapper, AmplitudeTransformerWrapper, AmplitudeGATrWrapper
from experiments.baselines import MLP, BaselineTransformer
from experiments.amplitudes.dataset import AmplitudeDataset
from experiments.amplitudes.preprocessing import preprocess_particles, preprocess_amplitude, undo_preprocess_amplitude
from experiments.amplitudes.plots import plot_histograms, plot_loss, plot_single_histogram
from gatr.nets import GATr
from gatr.layers import MLPConfig, SelfAttentionConfig

import matplotlib.pyplot as plt # debugging

class AmplitudeExperiment:

    def __init__(self, params):
        self.params = params
        self.device = self.params.get("device", get_device())
        
        self.warm_start = self.params.get("warm_start", False)
        self.warm_start_path = self.params.get("warm_start_path", None) if self.warm_start else None

        self.batch_size = self.params.get("batch_size", 1024)
        self.epoch = self.params.get("total_epochs", 0)
        self.runs = self.params.get("runs", 0) +1
        self.params["runs"] = self.runs

        self.bayesian = self.params.get("bayesian", False)
        self.bayesian_iterations = self.params.get("bayesian_iterations", 1)

        self.train = self.params.get("train", True)
        self.plot = self.params.get("plot", True)
 
        self.starttime = time.time()

    def prepare_experiment(self):
            
        if self.warm_start:
            assert self.warm_start_path is not None, \
                f"prepare_experiment: warm_start set to True, but warm_start_path not specified"
            assert os.path.exists(self.warm_start_path), \
                f"prepare_experiment: warm_start set to True, but warm_start_path {self.warm_start_path} does not exist"
            self.out_dir = self.warm_start_path
            os.chdir(self.out_dir)

        else:
            runs_dir = self.params.get("runs_dir", None)
            if runs_dir is None:
                runs_dir = os.path.join(os.getcwd(), "runs")
                print("prepare_experiment: runs_dir not specified. Working in ", runs_dir)
            run_name = self.params.get("run_name", None)
            rnd_number = np.random.randint(low=1000, high=9999)
            if run_name is None:
                self.out_dir = os.path.join(runs_dir, str(rnd_number))
                print("prepare_experiment: run_name not specified. Using random number")
            else:
                self.out_dir = os.path.join(runs_dir, run_name + str(rnd_number))
            os.makedirs(self.out_dir)
            os.chdir(self.out_dir)
            os.makedirs("models", exist_ok=True)
            
        self.params["out_dir"] = self.out_dir
        save_config(self.params, f"config.yaml")
        save_config(self.params, f"config{self.runs}.yaml")

        print(f"prepare_experiment: Using out_dir {self.out_dir} and device {self.device} for run {self.runs}")
        
        # redirect console (useful when working on cluster)
        if self.params.get("redirect_console", True):
            sys.stdout = open(f"stdout{self.runs}.txt", "w", buffering=1)
            sys.stderr = open(f"stderr{self.runs}.txt", "w", buffering=1)
            print(f"prepare_experiment: Redirecting console output to {self.out_dir}")

        # save source code
        if self.params.get("save_source", True):
            zip_name = "source.zip"
            code_dir = "../../gatr"
            zipf = zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED)
            for root, dirs, files in os.walk(code_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, code_dir))
            zipf.close()
            print(f"prepare_experiment: Saved source code to {zip_name}")

    def load_data(self):
        # load data
        self.dataset = self.params["dataset"]
        assert self.dataset in ["aag", "aagg", "zjj", "zjjj", "zjjjj"]

        self.type_token_dict = {"aag": [0,1,2,2,3], "aagg": [0,1,2,2,3,3],
                           "zjj": [0,1,2,3,3], "zjjj": [0,1,2,3,3,3],
                           "zjjjj": [0,1,2,3,3,3,3]}

        data_path = self.params["data_path"]
        dataset_path = os.path.join(data_path, f"{self.dataset}.npy")
        assert os.path.exists(dataset_path), f"path {dataset_path} does not exist"
        data_raw = np.load(dataset_path)
        self.particles = data_raw[:,:-1]
        self.amplitudes = data_raw[:,-1]

        # preprocess data
        self.amplitudes_prepd, self.amplitudes_mean, self.amplitudes_std = preprocess_amplitude(self.amplitudes)
        self.particles_prepd, self.particles_mean, self.particles_std = preprocess_particles(self.particles)

    def build_model(self):
        # load model
        model_type = self.params.get("model", None)
        if model_type is None:
            raise ValueError("build_model: model not specified")
        
        if model_type == "mlp":
            self.type_token = None
            
            in_shape = self.particles.shape[1]
            out_shape = (1,)
            hidden_channels = self.params.get("hidden_channels", 32)
            hidden_layers = self.params.get("hidden_layers", 4)
            net = MLP(in_shape=in_shape, out_shape=out_shape, hidden_channels=hidden_channels, hidden_layers=hidden_layers)
            self.model = AmplitudeMLPWrapper(net, self.particles_mean, self.particles_std)
            
        elif model_type == "transformer":
            self.type_token = self.type_token_dict[self.dataset]
            
            in_channels = 4 + (max(self.type_token) + 1)
            out_channels = 1
            hidden_channels = self.params.get("hidden_channels", 16)
            num_blocks = self.params.get("num_blocks", 1)
            num_heads = self.params.get("num_heads", 2)
            increase_hidden_channels = self.params.get("increase_hidden_channels", 4)
            multi_query = self.params.get("multi_query", False)
            net = BaselineTransformer(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels,
                                      num_blocks=num_blocks, num_heads=num_heads, pos_encoding=False,
                                      increase_hidden_channels=increase_hidden_channels, multi_query=multi_query)
            self.model = AmplitudeTransformerWrapper(net, self.particles_mean, self.particles_std)
        elif model_type == "gatr":
            self.type_token = self.type_token_dict[self.dataset]
            
            in_mv_channels = 1
            out_mv_channels = 1
            hidden_mv_channels = self.params.get("hidden_mv_channels", 1)
            in_s_channels = max(self.type_token) + 1
            out_s_channels = 1
            hidden_s_channels = self.params.get("hidden_s_channels", 8)
            num_blocks = self.params.get("num_blocks", 1)
            dropout_prob = self.params.get("dropout_prob", 0.0)

            multi_query = self.params.get("multi_query", True)
            num_heads = self.params.get("num_heads", 2)
            increase_hidden_channels = self.params.get("increase_hidden_channels", 2)
            attention = SelfAttentionConfig(multi_query=multi_query,
                                            in_mv_channels=hidden_mv_channels, out_mv_channels=hidden_mv_channels,
                                            in_s_channels=hidden_s_channels, out_s_channels=hidden_s_channels,
                                            num_heads=num_heads, increase_hidden_channels=increase_hidden_channels)
            
            mlp = MLPConfig(mv_channels=[hidden_mv_channels, 2*hidden_mv_channels, hidden_mv_channels],
                            s_channels=[hidden_s_channels, 2*hidden_s_channels, hidden_s_channels],
                            activation=self.params.get("activation", "gelu"))
            
            net = GATr(in_mv_channels=in_mv_channels, out_mv_channels=out_mv_channels, hidden_mv_channels=hidden_mv_channels,
                       in_s_channels=in_s_channels, out_s_channels=out_s_channels, hidden_s_channels=hidden_s_channels,
                       num_blocks=num_blocks, dropout_prob=dropout_prob, mlp=mlp, attention=attention)
            self.model = AmplitudeGATrWrapper(net)
            
        else:
            raise ValueError(f"build_model: model class {model_type} not recognised. Use INN, CFM, DDPM or DAT")
        model_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.params["model_parameters"] = model_parameters
        print(f"build_model: Built model {model_type}. Total number of parameters: {model_parameters}")

        # load pretrained model if warm_start
        if self.warm_start:
            try:
                model_idx = self.params.get("model_idx", 1)
                model_name = self.params.get("model_name", f"model_run{model_idx}")
                model_path = self.warm_start_path + f"/models/{model_name}.pt"
                assert os.path.exists(model_path), f"No model saved under {model_path}, probably there was no training yet in the experiment."
                state_dict = torch.load(model_path, map_location=self.device)
            except FileNotFoundError:
                raise ValueError(f"build_model: cannot load model {model_name} from {self.warm_start_path}")
            self.model.load_state_dict(state_dict)
            print(f"build_model: Loaded state_dict of model {model_name} from path {self.warm_start_path}")

    def build_optimizer(self):
        if self.train:
            optim = self.params.get("optimizer", "Adam")
            if optim == "Adam":
                lr = self.params.get("lr", 0.0001)
                betas = self.params.get("betas", [0.9, 0.999])
                weight_decay = self.params.get("weight_decay", 0)
                eps=self.params.get("adam_eps", 1.e-8)
                self.optimizer = \
                    AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay,
                             eps=eps) #always AdamW instead of Adam (only differs for weight_decay)
                print(
                    f"build_optimizer: Built optimizer {optim} with lr {lr}, betas {betas}, weight_decay {weight_decay}")
            else:
                raise ValueError(f"build_optimizer: optimizer {optim} not implemented")
        else:
            self.optimizer = None
            print("build_optimizer: train set to False. Not building optimizer")

    def build_dataloaders(self):
        n_data = len(self.particles)
        data_split = self.params.get("data_split", 0.5)
        self.cut = int(n_data * data_split)

        # BNN: add #trainingdata here
            
        self.train_loader = \
            DataLoader(dataset=AmplitudeDataset(self.particles[:self.cut], self.amplitudes_prepd[:self.cut]),
                           batch_size=self.batch_size,
                           shuffle=True)
        self.test_loader = \
            DataLoader(dataset=AmplitudeDataset(self.particles[self.cut:], self.amplitudes_prepd[self.cut:]),
                           batch_size=self.batch_size,
                           shuffle=False)
        print(f"build_dataloaders: Built dataloaders with data_split {data_split} and batch_size {self.batch_size}")

    def build_lrscheduler(self):
        if self.train:
            self.lr_scheduler = self.params.get("lr_scheduler", None)
            if self.lr_scheduler == "OneCycle":
                lr = self.params.get("lr", 0.0001)
                n_epochs = self.params["n_epochs"]
                self.scheduler = OneCycleLR(
                    self.optimizer,
                    lr * 10,
                    epochs=n_epochs,
                    steps_per_epoch=len(self.train_loader))
                print("build_dataloaders: Using one-cycle lr scheduler")
            elif self.lr_scheduler == "CosineAnnealing":
                n_epochs = self.params["n_epochs"]
                eta_min = self.params.get("eta_min", 0)
                self.scheduler = CosineAnnealingLR(
                    optimizer=self.optimizer,
                    T_max=n_epochs*len(self.train_loader),
                    eta_min=eta_min
                )
                print(f"build_dataloaders: Using CosineAnnealing lr scheduler with eta_min {eta_min}")
            elif self.lr_scheduler is None:
                print(f"build_dataloaders: Using constant learning rate.")
                self.scheduler = None
            else:
                print(f"build_dataloaders: lr_scheduler {self.lr_scheduler} not recognised. Defaulting to constant learning rate")
                self.scheduler = None
        else:
            print("build_dataloaders: train set to False. Not building lr scheduler")

    def train_model(self):
        if self.train:
            self.run_training()
            torch.save(self.model.state_dict(), f"models/model_run{self.runs}.pt")
            
            print(f"train_model: Model has been trained for a total of {self.epoch} epochs")
            self.params["total_epochs"] = self.epoch
        else:
            print("train_model: train set to False. Not training")
    
    def run_training(self):
        self.train_losses = np.array([])
        self.train_lr = np.array([])
        self.loss_fn = torch.nn.MSELoss()
        n_epochs = self.params["n_epochs"]
        print(f"train_model: Model has been trained for {self.epoch} epochs before.")
        print(f"train_model: Beginning training for n_epochs={n_epochs}")
        t0 = time.time()
        for e in range(n_epochs):
            self.epoch += 1
            self.model.train()
            self.train_one_epoch()

            if e==0:
                t1 = time.time()
                dtEst= (t1-t0) * n_epochs
                print(f"Training time estimate: {dtEst/60:.2f} min = {dtEst/60**2:.2f} h")

        t1 = time.time()
        traintime = t1 - t0
        self.params["traintime"] = traintime
        print(f"train_model: Finished training for {n_epochs} epochs after {traintime:.2f} s = {traintime/60:.2f} min = {traintime/60**2:.2f} h.")

    def train_one_epoch(self):
        train_losses = []
        train_lr = []
        self.optimizer.zero_grad()
        for batch_id, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x, type_token=self.type_token).flatten()
            loss = self.loss_fn(y, y_pred)
            #if self.epoch == 4:
            #    print( torch.mean((y - y_pred)**2).item(), loss.item())

            if torch.isfinite(loss).all():
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.lr_scheduler is not None:
                    self.scheduler.step()
                    
                train_losses.append(loss.item())
                train_lr.append(self.optimizer.param_groups[0]["lr"])
            else:
                print(f"train_model: Unstable loss. Skipped backprop for epoch {self.epoch}, batch_id {batch_id}")

        self.train_losses = np.concatenate([self.train_losses, train_losses], axis=0)
        self.train_lr = np.concatenate([self.train_lr, train_lr], axis=0)

    def evaluate_model(self):
        # compute predictions for amplitudes
        self.amplitudes_truth_prepd = self.test_loader.dataset.amplitudes.numpy()
        self.amplitudes_prediction_prepd = np.zeros(0)
        for x, y in self.test_loader:
            y_pred = self.model(x, type_token=self.type_token).flatten().detach().cpu().numpy()
            self.amplitudes_prediction_prepd = np.concatenate((self.amplitudes_prediction_prepd, y_pred), axis=0)
        assert self.amplitudes_truth_prepd.shape == self.amplitudes_prediction_prepd.shape, \
               f"{self.amplitudes_truth_prepd.shape} != {self.amplitudes_prediction_prepd.shape}"

        # compute mse over test dataset
        self.mse = np.mean( (self.amplitudes_truth_prepd - self.amplitudes_prediction_prepd) ** 2)
        print(f"MSE on test dataset: {self.mse:.4f}")

        # undo preprocessing for amplitudes
        self.amplitudes_truth = undo_preprocess_amplitude(self.amplitudes_truth_prepd,
                                                          self.amplitudes_mean, self.amplitudes_std)
        self.amplitudes_prediction = undo_preprocess_amplitude(self.amplitudes_prediction_prepd,
                                                          self.amplitudes_mean, self.amplitudes_std)

    def plot_results(self):
        if self.plot:
            print(f"make_plots: Start making plots")
            self._plot()

            print("make_plots: Finished making plots")
        else:
            print("make_plots: No plotting because plot=False")

    def _plot(self):
        path = f"plots/run{self.runs}"
        os.makedirs(path, exist_ok=True)
        params_plot = self.params.get("params_plot", {})
        self.dataset_title = {"aag": r"$\gamma\gamma g$", "aagg": r"$\gamma\gamma gg$",
                              "zjj": "$Zjj$", "zjjj": "$Zjjj$", "zjjjj": "$Zjjjj$"}[self.dataset]
        
        if params_plot.get("plot_loss", True):
            file = f"{path}/loss.pdf"
            plot_loss(file, [self.train_losses], self.train_lr, labels=["loss"])

        if params_plot.get("plot_histograms", True):
            out = f"{path}/histograms.pdf"
            with PdfPages(out) as file:
                labels = ["Test", "Train", "Prediction"]

                func = lambda x: np.log(x)
                data = [func(self.amplitudes_truth), func(self.amplitudes[:self.cut]), func(self.amplitudes_prediction)]
                plot_histograms(file, data, labels, title=self.dataset_title,
                           xlabel=r"$\log A$", logx=False)

        if params_plot.get("plot_delta", True):
            out = f"{path}/delta.pdf"
            with PdfPages(out) as file:
                data = (self.amplitudes_truth - self.amplitudes_prediction) / self.amplitudes_truth
                plot_single_histogram(file, data, title=self.dataset_title,
                           xlabel=r"$\Delta = \frac{A_\mathrm{truth} - A_\mathrm{pred}}{A_\mathrm{truth}}$",
                           logx=False, xrange=(-.3, .3), bins=50)
            

    def finish_up(self):
        self.params["datetime"] = str(datetime.now())
        dt = time.time() - self.starttime
        self.params["experimenttime"] = dt
        save_config(self.params, f"config{self.runs}.yaml")
        
        print(f"finish_up: Finished experiment after {dt:.2f} s = {dt/60:.2f} min = {dt/60**2:.2f} h")

        if self.params.get("redirect_console", True):
            sys.stdout.close()
            sys.stderr.close()

    def full_run(self):
        self.prepare_experiment()
        self.load_data()
        self.build_model()
        self.build_optimizer()
        self.build_dataloaders()
        self.build_lrscheduler()
        self.train_model()
        self.evaluate_model()
        self.plot_results()
        self.finish_up()

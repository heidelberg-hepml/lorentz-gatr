#!/usr/bin/env python3

import os
import hydra
import optuna
import torch
from hydra import compose, initialize
from optuna import Trial
from pathlib import Path

from experiments.toptagging.experiment import TopTaggingExperiment


def run_trial(trial: Trial, seed, exp_name):
    """Performs a trial: samples hyperparameters, trains a model, and returns the validation error"""

    # Choose trial params
    num_blocks = trial.suggest_int("blocks", 4, 32)
    hidden_mv_channels = 2 ** trial.suggest_int("mv_channels", 2, 4)
    hidden_s_channels = 2 ** trial.suggest_int("s_channels", 4, 7)
    num_heads = 2 ** trial.suggest_int("heads", 2, 4)
    multi_query = trial.suggest_categorical("multi_query", ["true", "false"])
    increase_hidden_channels = 2 ** trial.suggest_int("increase_hidden_channels", 0, 1)
    beam_reference = trial.suggest_categorical(
        "beam_reference", ["null", "photon", "spacelike", "xyplane"]
    )

    # catch runs with invalid hyperparameters
    if num_heads > increase_hidden_channels * hidden_mv_channels:
        raise optuna.TrialPruned()

    with initialize(config_path="config", version_base=None):
        overrides = [
            # optuna-related settings
            f"run_name=trial_{trial.number}",
            f"exp_name={exp_name}",
            # f"seed={seed}",
            # Fixed settings
            "training.iterations=10000",
            "training.batchsize=128",
            "training.scheduler=null",
            "training.lr=1e-4",
            "training.force_xformers=true",
            # Tuned parameters
            f"model.net.num_blocks={num_blocks}",
            f"model.net.hidden_mv_channels={hidden_mv_channels}",
            f"model.net.hidden_s_channels={hidden_s_channels}",
            f"model.net.attention.num_heads={num_heads}",
            f"model.net.attention.multi_query={multi_query}",
            f"model.net.attention.increase_hidden_channels={increase_hidden_channels}",
            f"model.beam_reference={beam_reference}",
        ]
        cfg = compose(config_name="toptagging", overrides=overrides)
        try:
            exp = TopTaggingExperiment(cfg)
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            print("Pruning trial {trial.number} due to torch.cuda.OutOfMemoryError")
            raise optuna.TrialPruned()

        # Run experiment
        exp()
        score = exp.results["val"]["rej05"]  # use accuracy as score

        return score


# add sweep configuration to the basic cfg -> Have optuna.db in same directory as everything else
@hydra.main(config_path="config", config_name="toptagging", version_base=None)
def sweep(cfg):
    """Entrance point to parameter sweep (wrapped with hydra)"""

    # Clear hydra instances
    # Important so we can use hydra again in the experiment
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    # Create or load study
    Path(cfg.sweep.optuna_db).parent.mkdir(exist_ok=True, parents=True)
    study = optuna.create_study(
        storage=f"sqlite:///{Path(cfg.sweep.optuna_db).resolve()}?timeout=60",
        load_if_exists=True,
        study_name=cfg.exp_name,
        direction="maximize",
    )

    # Let's go
    study.optimize(
        lambda trial: run_trial(trial, seed=cfg.sweep.seed, exp_name=cfg.exp_name), n_trials=cfg.sweep.trials
    )


if __name__ == "__main__":
    sweep()

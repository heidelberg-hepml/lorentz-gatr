import numpy as np


def preprocess1(events, prep_params=None):
    if prep_params is None:
        prep_params = {"std": events.std()}
    events = events / prep_params["std"]
    return events, prep_params


def undo_preprocess1(events, prep_params):
    events = events * prep_params["std"]
    return events

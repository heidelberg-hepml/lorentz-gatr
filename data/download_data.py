import numpy as np
import h5py
import hdf5plugin
import os, sys
import wget

base_url = "https://www.thphys.uni-heidelberg.de/~plehn/data"
files_to_collect = ["amplitudes", "toptagging", "event-generation"]

download = {"amplitudes": True,
            "toptagging": True,
            "event-generation": True}
urls = {"amplitudes": "amplitudes.hdf5",
        "toptagging": "toptagging_full.npz",
        "event-generation": "event_generation_ttbar.hdf5"}

def load(filename):
    url = os.path.join(base_url, filename)
    print(f"Started to download {filename}")
    wget.download(url, filename)
    print(f"Successfully downloaded {filename}")

if "amplitudes" in files_to_collect:
    filename = urls["amplitudes"]
    if download["amplitudes"]:
        load(filename)
    with h5py.File(filename, "r") as f:
        for key in ["zg", "zgg", "zggg", "zgggg"]:
            data = f[key]
            np.save(f"{key}.npy", data)
            print(f"Successfully created {key}.npy")

if "toptagging" in files_to_collect:
    filename = urls["toptagging"]
    if download["toptagging"]:
        load(filename)

if "event-generation" in files_to_collect:
    filename = urls["event-generation"]
    if download["event-generation"]:
        load(filename)
    with h5py.File(filename, "r") as f:
        for njets in range(5):
            data = f[f"ttbar+{njets}j"]
            np.save(data, f"ttbar_{njets}j.npy")
            print(f"Successfully created ttbar_{njets}j.npy")

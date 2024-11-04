import numpy as np
import h5py
import hdf5plugin
import os
import wget

# specify here which datasets you want to download
# dataset sizes: amplitudes 0.6G, toptagging 1.5G, event-generation 4.7G
DOWNLOAD = {"amplitudes": True, "toptagging": True, "event-generation": True}
# and which you want to unzip
UNZIP = {"amplitudes": True, "toptagging": True, "event-generation": True}

BASE_URL = "https://www.thphys.uni-heidelberg.de/~plehn/data"
FILENAMES = {
    "amplitudes": "amplitudes.hdf5",
    "toptagging": "toptagging_full.npz",
    "event-generation": "event_generation_ttbar.hdf5",
}
DATA_DIR = "data"

"""
To use the JetClass dataset, download all .tar files from 
https://zenodo.org/records/6619768 (around 100G),
unzip them, e.g. with 'tar -xvf *.tar', 
modify the data.data_dir in jctagging.yaml 
to point to the directory containing the train_100M, test_20M, val_5M folders.
"""


def load(filename):
    url = os.path.join(BASE_URL, filename)
    print(f"Started to download {url}")
    target_path = os.path.join(DATA_DIR, filename)
    wget.download(url, out=target_path)
    print("")
    print(f"Successfully downloaded {target_path}")


def main():
    # collect amplitudes dataset
    # we created this dataset ourselves, see paper for details
    filename = FILENAMES["amplitudes"]
    if DOWNLOAD["amplitudes"]:
        load(filename)
    if UNZIP["amplitudes"]:
        filename = os.path.join(DATA_DIR, filename)
        with h5py.File(filename, "r") as file:
            for key in ["zg", "zgg", "zggg", "zgggg"]:
                data = file[key]
                target_path = os.path.join(DATA_DIR, f"{key}.npy")
                np.save(target_path, data)
                print(f"Successfully created {target_path}")

    # collect toptagging dataset
    # this is a npz version of the original dataset at https://zenodo.org/records/2603256
    filename = FILENAMES["toptagging"]
    if DOWNLOAD["toptagging"]:
        load(filename)

    # collect event generation dataset
    # we created this dataset ourselves, see paper for details
    filename = FILENAMES["event-generation"]
    if DOWNLOAD["event-generation"]:
        load(filename)
    if UNZIP["event-generation"]:
        filename = os.path.join(DATA_DIR, filename)
        with h5py.File(filename, "r") as file:
            for njets in range(5):
                data = file[f"ttbar+{njets}jet"]
                target_path = os.path.join(DATA_DIR, f"ttbar_{njets}j.npy")
                np.save(data, target_path)
                print(f"Successfully created {target_path}")


if __name__ == "__main__":
    main()

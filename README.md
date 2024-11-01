# Lorentz-Equivariant Geometric Algebra Transformer

This repository contains the official implementation of the [**Lorentz-Equivariant Geometric Algebra Transformer**](https://arxiv.org/abs/2405.14806) by [Jonas Spinner](mailto:j.spinner@thphys.uni-heidelberg.de), [Víctor Bresó](mailto:breso@thphys.uni-heidelberg.de), Pim de Haan, Tilman Plehn, Jesse Thaler, and Johann Brehmer.

![](img/gatr.png)


## 1. Getting started

Clone the repository
```bash
git clone https://github.com/heidelberg-hepml/lorentz-gatr
```

Create a virtual environment and install requirements
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

In case you want to run our experiments, you first have to collect the datasets. They can be downloaded from the Heidelberg ITP website ([amplitudes](https://www.thphys.uni-heidelberg.de/~plehn/data/amplitudes.hdf5), [toptagging](https://www.thphys.uni-heidelberg.de/~plehn/data/toptagging_full.npz), [event-generation](https://www.thphys.uni-heidelberg.de/~plehn/data/event_generation_ttbar.hdf5)). hdf5 archives have to be unpacked to npy files for each key in the archive. Finally, keys in the `data` section of the config files have to be adapted to specify where the datasets are located on your machine (`data_path` or `data_dir` depending on the experiment). The following command automates this procedure, and modifying the script allows you to collect only some datasets
```bash
python data/collect_data.py
```

## 2. Running experiments

You can run any of our experiments with the following commands:
```bash
python run.py -cn amplitudes model=gatr_amplitudes exp_name=amplitudes run_name=hello_world_amplitudes
python run.py -cn toptagging model=gatr_toptagging exp_name=toptagging run_name=hello_world_toptagging
python run.py -cn ttbar model=gatr_eventgen exp_name=eventgen run_name=hello_world_eventgen
```

We use hydra for configuration management, allowing to quickly override parameters in e.g. config/amplitudes.yaml. Further, we use mlflow for tracking. You can start a mlflow server based on the saved results in runs/tracking/mlflow.db on port 4242 of your machine with the following command, executed from a directory that contains the `runs` folder.

```bash
mlflow ui --port 4242 --backend-store-uri sqlite:///runs/tracking/mlflow.db
```

An existing run can be reloaded to perform additional tests with the trained model. For a previous run with exp_name=amplitudes and run_name=hello_world_amplitudes, one can run for example. 

```bash
python run.py -cn config -cp runs/amplitudes/hello_world_amplitudes train=false warm_start_idx=0
```

The warm_start_idx specifies which model in the models folder should be loaded and defaults to 0. 

The default configuration files in the `config` folder define small models to allow quick test runs. If you want to reproduce the longer experiments in the paper, you can use the configuration files in `config_paper`.

## 3. Using L-GATr 

To use L-GATr on your own problem, you will at least need two components from this repository:
L-GATr networks, which act on multivector data, and interface functions that embed various geometric
objects into this multivector representations.

Here is an example code snippet that illustrates the recipe:

```python
from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_vector, extract_scalar
import torch


class ExampleWrapper(torch.nn.Module):
    """Example wrapper around a L-GATr model.
    
    Expects input data that consists of a point cloud: one 4-momentum point for each item in the data.
    Returns outputs that consists of one scalar number for the whole dataset.
    
    Parameters
    ----------
    blocks : int
        Number of transformer blocks
    hidden_mv_channels : int
        Number of hidden multivector channels
    hidden_s_channels : int
        Number of hidden scalar channels
    """

    def __init__(self, blocks=10, hidden_mv_channels=16, hidden_s_channels=32):
        super().__init__()
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),  # Use default parameters for attention
            mlp=MLPConfig(),  # Use default parameters for MLP
        )
        
    def forward(self, inputs):
        """Forward pass.
        
        Parameters
        ----------
        inputs : torch.Tensor with shape (*batch_dimensions, num_points, 4)
            4-momentum point cloud input data
        
        Returns
        -------
        outputs : torch.Tensor with shape (*batch_dimensions, 1)
            Model prediction: a single scalar for the whole point cloud.
        """
        
        # Embed 4-momentum point cloud inputs in GA
        embedded_inputs = embed_vector(inputs).unsqueeze(-2)  # (..., num_points, 1, 16)
        
        # Pass data through GATr
        embedded_outputs, _ = self.gatr(embedded_inputs, scalars=None)  # (..., num_points, 1, 16)
        
        # Extract scalar outputs 
        outputs = extract_scalar(embedded_outputs)  # (..., 1)

        return outputs
```

In the following, we will go into more detail on the conventions used in this code base and the
structure of the repository.

### Design choices

**Representations**: L-GATr operates with two kind of representations: geometric algebra multivectors
and auxiliary scalar representations. Both are simply represented as `torch.Tensor` instances.

The multivectors are based on the geometric algebra Cl(1, 3). They are tensors of the
shape `(..., 16)`, for instance `(batchsize, items, channels, 16)`. The sixteen multivector
components are sorted as in the
[`clifford` library](https://clifford.readthedocs.io/en/latest/), as follows:
`[x_scalars, x_0, x_1, x_2, x_3, x_01, x_02, x_03, x_12, x_13, x_23, x_012, x_013, x_023, x_123,
x_0123]`.

Scalar representations have free shapes, but should match the multivector representations they
accompany in batchsize and number of items. The number of channels may be different.

**Functions**: We distinguish between primitives (functions) and layers (often stateful
`torch.nn.Module` instances). Almost all primitives and layers are Pin(1, 3)-equivariant,
see docstrings for exceptions.

### Repository structure

```text
lorentz-gatr
|
└───config: configuration YAML files for the experiments, with small models and few iterations to quickly test the code
|   └───model: model configurations
|   └───classifier: classifier metric configuration (event generation experiment)
|   |   amplitudes.yaml: configuration for the amplitude experiment
|   |   default.yaml: default configuration
|   |   hydra.yaml: hydra configuration
|   |   default_tagging.yaml: default configuration for tagging experiments
|   |   toptagging.yaml: configuration for the top-tagging experiment
|   |   jctagging.yaml: configuration for the JetClass tagging experiment
|   |   qgtagging.yaml: configuration for the quark-gluon-tagging experiment
|   |   ttbar.yaml: configuration for the ttbar event-generation experiment
|   |   zmumu.yaml: configuration for the z->mumu event-generation experiment
|
└───config_paper: configuration YAML files for the experiments, with the hyperparameters used in the paper
|   └───model: model configurations
|   └───classifier: classifier metric configuration (event generation experiment)
|   |   amplitudes.yaml: configuration for the amplitude experiment
|   |   default.yaml: default configuration
|   |   hydra.yaml: hydra configuration
|   |   toptagging.yaml: configuration for the toptagging experiment
|   |   ttbar.yaml: configuration for the ttbar event-generation experiment
|
└───data: space to store datasets
|   |   download_data.py: download and unpack datasets
|
└───experiments: experiments that use gatr
|   └───baselines: baseline layers and architectures
|   └───amplitudes: amplitude experiment
|   └───tagging: tagging experiments
|   └───eventgen: event generation experiments
|   |
|   |   misc.py: various utility functions
|   |   logger.py: Logger setup
|   |   mlflow.py: MLFlow logger 
|   |   base_experiment.py: Base class for all experiments including model, optimizer and scheduler initialization, logging protocol and training/validation loops
|   |   base_plots.py: Example plot functions (not used)
|   |   base_wrapper.py: Example wrapper for L-GATr (not used) 
│
└───gatr: core library
|   └───interface: embedding of geometric quantities into projective geometric algebra
|   |   |   vector.py: Lorentz vectors
|   |   |   scalar.py: scalars
|   |   |   spurions.py: spurions for symmetry breaking
|   |
|   └───layers: network layers
|   |   └───attention: self-attention layer, its components, and the corresponding configuration
|   |   └───mlp: geometric MLP, its components, and the corresponding configuration
|   |   |   dropout.py: multivector dropout
|   |   |   gatr_block.py: L-GATr transformer block, the main layer that L-GATr networks consist of
|   |   |   layer_norm.py: geometric LayerNorm
|   |   |   linear.py: equivariant linear layer between multivectors
|   |
|   └───nets: complete network architectures
|   |   |   axial_gatr.py: axial-attention version of L-GATr for two token dimensions
|   |   |   gatr.py: L-GATr architecture for a single token dimension
|   |   |   gap.py: L-GATr architecture without the transformer module
|   |
|   └───primitives: core functional building blocks of L-GATr
|   |   |   attention.py: geometric attention mechanism
|   |   |   bilinear.py: bilinear equivariant functions like the geometric product
|   |   |   dropout.py: geometric dropout
|   |   |   invariants.py: invariant functions of multivectors like the norm
|   |   |   linear.py: equivariant linear maps between multivectors
|   |   |   nonlinearities.py: gated nonlinearities
|   |   |   normalization.py: geometric normalization functions
|   |
|   └───utils: utility functions
|       |   clifford.py: non-differentiable GA functions based on the clifford library
|       |   einsum.py: optimized einsum function
|       |   tensors.py: various tensor operations
|
└───img: images
│
└───tests: unit tests (e.g. for self-consistency and Pin equivariance)
|   └───gatr
|   |   └───interface: unit tests for gatr.interface
|   |   └───layers: unit tests for gatr.layers
|   |   └───nets: unit tests for gatr.nets
|   |   └───primitives: unit tests for gatr.primitives
|   |   └───utils: unit tests for gatr.utils
|   |
|   └───helpers: utility functions for unit tests
|   |   |   constants.py: test settings (like tolerances)
|   |   |   equivariance.py: helper functions to test Pin equivariance
|   |   |   geometric_algebra.py: helper functions to test GA functionality
|   |
|   └───experiments
|   |   └───eventgen: unit tests for base distributions and transforms/coordinates classes
│
└───tests_regression: regression tests
│  
|   .gitignore: git configuration
│   LICENSE: license under which this code may be used
|   pyproject.toml: project settings
│   README.md: this README file
|   requirements.txt: external dependencies
```

## 4. Extra features in this repository

Here we list some additional functional elements of the code that are not explicitly mentioned in the paper:

1. Tagging experiment for a quark gluon dataset containing extra scalar features
2. Extra options in the tagging experiment to include more scalar variables, mean aggregation etc
3. Extra base distributions and variable parametrizations for event generation
4. Event generation experiment for Z + jets
5. Features of the original GATr repo that we do not use: Positional encodings, axial transformer and axial L-GATr build

## 5. Citation

If you find our code useful, please cite:

```text
@article{Spinner:2024hjm,
    author = "Spinner, Jonas and Bres\'o, Victor and de Haan, Pim and Plehn, Tilman and Thaler, Jesse and Brehmer, Johann",
    title = "{Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics}",
    eprint = "2405.14806",
    archivePrefix = "arXiv",
    primaryClass = "physics.data-an",
    reportNumber = "MIT-CTP/5723",
    month = "5",
    year = "2024"
}
```

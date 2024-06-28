# Lorentz-Equivariant Geometric Algebra Transformer

This repository contains the official implementation of the [**Lorentz-Equivariant Geometric Algebra Transformer**](https://arxiv.org/abs/2405.14806) by [Jonas Spinner](mailto:j.spinner@thphys.uni-heidelberg.de), [Víctor Bresó](mailto:breso@thphys.uni-heidelberg.de), Pim de Haan, Tilman Plehn, Jesse Thaler, and Johann Brehmer.

## 1. Getting started

Clone the repository.

```bash
git clone https://github.com/heidelberg-hepml/lorentz-gatr
```

Create a virtual environment and install requirements

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The datasets can be downloaded from the Heidelberg ITP website ([amplitudes](https://www.thphys.uni-heidelberg.de/~plehn/data/amplitudes.hdf5), [toptagging](https://www.thphys.uni-heidelberg.de/~plehn/data/toptagging_full.npz), [event-generation](https://www.thphys.uni-heidelberg.de/~plehn/data/event_generation_ttbar.hdf5)). hdf5 archives have to be unpacked into npy files for each key in the archive. The script lorentz-gatr/data/download_data.py can be used to download the data. Finally, adapt the keys in the `data` section of the config files to specify where the datasets are located on your machine (`data_path` or `data_dir` depending on the experiment).

## 2. Running experiments

You can run any of our experiments with the following commands:
```bash
python run.py model=gatr_amplitudes exp_type=amplitudes exp_name=amplitudes run_name=hello_world_amplitudes
python run.py model=gatr_toptagging exp_type=toptagging exp_name=toptagging run_name=hello_world_toptagging
python run.py model=gatr_eventgen exp_type=ttbar exp_name=eventgen run_name=hello_world_eventgen
```

We use hydra for configuration management, allowing to quickly override parameters in e.g. config/amplitudes.yaml. Further, we use mlflow for tracking. You can start a mlflow server based on the saved results in runs/tracking/mlflow.db on port 4242 of your machine with the following command

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

    def __init__(self, blocks=20, hidden_mv_channels=16, hidden_s_channels=32):
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
└───config: configuration YAML files for the experiments
|   └───model: model configurations
|   └───classifier: classifier metric configuration (event generation experiment)
|   |   amplitudes.yaml: configuration for the amplitude experiment
|   |   default.yaml: default configuration
|   |   hydra.yaml: hydra configuration
|   |   qgtagging.yaml: configuration for the quark-gluon-tagging experiment
|   |   toptagging.yaml: configuration for the toptagging experiment
|   |   ttbar.yaml: configuration for the ttbar event-generation experiment
|   |   z5g.yaml: configuration for the z+5g event-generation experiment
|   |   zmumu.yaml: configuration for the z->mumu event-generation experiment
|
└───data: space to store datasets
|   |   download_data.py: download and unpack datasets
└───gatr: core library
|   └───interface: embedding of geometric quantities into projective geometric algebra
|   |   |   vector.py: Lorentz vector
|   |   |   pseudoscalar.py: pseudoscalars (not used)
|   |   |   scalar.py: scalars
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
└───experiments: experiments that use gatr
|   └───baselines: baseline layers and architectures
|   |   |   mlp.py: MLP baseline
|   |   |   transformer.py: Transformer baseline
|   |   |   dsi.py: Deep Sets with Lorentz invariants baseline
|   |
|   └───amplitudes: amplitude experiment
|   |   |   dataset.py: data class builder
|   |   |   experiment.py: experiment class including dataloader definition, loss function and model evaluation
|   |   |   plots.py: plot builder
|   |   |   preprocessing.py: preprocessing functions for inputs and outputs
|   |   |   wrappers.py: wrapper classes for all baselines
|   └───toptagging: top tagging experiment
|   |   |   dataset.py: data class builder
|   |   |   experiment.py: experiment class including dataloader definition, loss function and model evaluation
|   |   |   plots.py: plot builder
|   |   |   wrappers.py: wrapper classes for all baselines
|   └───eventgen: event generation experiment
|   |   |   cfm.py: CFM base class for event generation
|   |   |   classifier.py: MLP classifier metric
|   |   |   coordinates.py: trajectory definitions and full transformation functions between coordinate spaces
|   |   |   distributions.py: base distributions
|   |   |   dataset.py: data class builder
|   |   |   experiment.py: experiment class including dataloader definition, loss function and model evaluation
|   |   |   helpers.py: helper functions for plotting
|   |   |   plots.py: base plotting functions
|   |   |   plotter.py: plot builder
|   |   |   processes.py: experiment settings for different physical processes
|   |   |   transforms.py: list of transformation functions between coordinate spaces
|   |   |   wrappers.py: wrapper class for a vector field networks
|   |
|   |   misc.py: various utility functions
|   |   logger.py: Logger setup
|   |   mlflow.py: MLFlow logger 
|   |   base_experiment.py: Base class for all experiments including model, optimizer and scheduler initialization, logging protocol and training/validation loops
|   |   base_plots.py: Example plot functions (not used)
|   |   base_wrapper.py: Example wrapper for L-GATr (not used) 
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
|       |   constants.py: test settings (like tolerances)
|       |   equivariance.py: helper functions to test Pin equivariance
|       |   geometric_algebra.py: helper functions to test GA functionality
│
|   |
|   └───experiments
|   |   └───eventgen: units tests for base distributions and transforms/coordinates classes
│
└───tests_regression: regression tests
│  
│   LICENSE: license under which this code may be used
│   README.md: this README file
|   requirements.txt: external dependencies
```

## 4. Extra features in this repository

Here we list some additional functional elements of the code that are not explicitly mentioned in the paper:

1. Axial transformer and axial L-GATr build
2. Tagging experiment for a quark gluon dataset containing extra scalar features
3. Extra options in the tagging experiment to include more scalar variables and particle pair information encoded as extra channels
4. Extra base distributions and variable parametrizations for event generation
5. Event generation experiments for Z + jets and Z + 5 gluons datasets

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

# Lorentz' Geometric Algebra Transformers

This repository contains an adapted version of the official implementation of the original Geometric Algebra Transformer of https://github.com/Qualcomm-AI-research/geometric-algebra-transformer for the Lorentz Group. 

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


## 2. Running experiments

TBD

## Using GATr (copied from the original GATr)

To use GATr on your own problem, you will at least need two components from this repository:
GATr networks, which act on multivector data, and interface functions that embed various geometric
objects into this multivector representations.

Here is an example code snippet that illustrates the recipe:

```python
from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_scalar
import torch


class ExampleWrapper(torch.nn.Module):
    """Example wrapper around a GATr model.
    
    Expects input data that consists of a point cloud: one 3D point for each item in the data.
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
        inputs : torch.Tensor with shape (*batch_dimensions, num_points, 3)
            Point cloud input data
        
        Returns
        -------
        outputs : torch.Tensor with shape (*batch_dimensions, 1)
            Model prediction: a single scalar for the whole point cloud.
        """
        
        # Embed point cloud in PGA
        embedded_inputs = embed_point(inputs).unsqueeze(-2)  # (..., num_points, 1, 16)
        
        # Pass data through GATr
        embedded_outputs, _ = self.gatr(embedded_inputs, scalars=None)  # (..., num_points, 1, 16)
        
        # Extract scalar and aggregate outputs from point cloud
        nodewise_outputs = extract_scalar(embedded_outputs)  # (..., num_points, 1, 1)
        outputs = torch.mean(nodewise_outputs, dim=(-3, -2))  # (..., 1)
        
        return outputs
```

In the following, we will go into more detail on the conventions used in this code base and the
structure of the repository.

### Design choices

**Representations**: GATr operates with two kind of representations: geometric algebra multivectors
and auxiliary scalar representations. Both are simply represented as `torch.Tensor` instances.

The multivectors are based on the projective geometric algebra G(3, 0, 1). They are tensors of the
shape `(..., 16)`, for instance `(batchsize, items, channels, 16)`. The sixteen multivector
components are sorted as in the
[`clifford` library](https://clifford.readthedocs.io/en/latest/), as follows:
`[x_scalars, x_0, x_1, x_2, x_3, x_01, x_02, x_03, x_12, x_13, x_23, x_012, x_013, x_023, x_123,
x_0123]`.

Scalar representations have free shapes, but should match the multivector representations they
accompany in batchsize and number of items. The number of channels may be different.

**Interface to the real world**: To map the multivector representations to physical objects, we
use the plane-based convention presented in
[Roelfs and De Keninck, "Graded symmetry groups: Plane and simple"](https://arxiv.org/abs/2107.03771).
3D points are thus represented as trivectors, planes as vectors, and so on. We provide these
these interface functions in the `gatr.interface` submodule.

**Functions**: We distinguish between primitives (functions) and layers (often stateful
`torch.nn.Module` instances). Almost all primitives and layers are Pin(3, 0, 1)-equivariant,
see docstrings for exceptions.

### Repository structure

```text
lorentz-gatr
|
└───config: configuration YAML files for the experiments
|   └───model: model configurations
|   |   amplitudes.yaml: default configuration for the amplitude experiment
|
└───gatr: core library
|   └───interface: embedding of geometric quantities into projective geometric algebra
|   |   |   vector.py: Lorentz vector
|   |   |   pseudoscalar.py: pseudoscalars
|   |   |   scalar.py: scalars
|   |
|   └───layers: network layers
|   |   └───attention: self-attention layer, its components, and the corresponding configuration
|   |   └───mlp: geometric MLP, its components, and the corresponding configuration
|   |   |   dropout.py: multivector dropout
|   |   |   gatr_block.py: GATr transformer block, the main layer that GATr networks consist of
|   |   |   layer_norm.py: geometric LayerNorm
|   |   |   linear.py: equivariant linear layer between multivectors
|   |
|   └───nets: complete network architectures
|   |   |   axial_gatr.py: axial-attention version of GATr for two token dimensions
|   |   |   gatr.py: GATr architecture for a single token dimension
|   |
|   └───primitives: core functional building blocks of GATr
|   |   └───data: pre-computed basis stored in `.pt` files
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
|       |   misc.py: various utility functions
|       |   tensors.py: various tensor operations
|
└───experiments: experiments that use gatr
|   └───baselines: baseline layers and architectures
|   |   |   mlp.py: MLP baseline
|   |   |   transformer.py: Transformer baseline
|   |
|   └───amplitudes: amplitude experiment
|   |
|   |   misc.py: various utility functions
│
└───tests: unit tests (e.g. for self-consistency and Pin equivariance)
|   └───gatr
|   |   └───baselines: unit tests for gatr.baselines
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
└───tests_regression: regression tests
│  
│   LICENSE: license under which this code may be used
│   README.md: this README file
|   requirements.txt: external dependencies
```

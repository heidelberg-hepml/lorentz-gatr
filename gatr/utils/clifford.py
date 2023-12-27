# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Geometric algebra operations based on the clifford library."""

from typing import Optional

import clifford
import numpy as np
import torch

LAYOUT, BLADES = clifford.Cl(1,3)

def np_to_mv(array):
    """Shorthand to transform a numpy array to a Pin(1,3) multivector."""
    return clifford.MultiVector(LAYOUT, value=array)


def tensor_to_mv(tensor):
    """Shorthand to transform a numpy array to a Pin(1,3) multivector."""
    return np_to_mv(tensor.detach().cpu().numpy())


def tensor_to_mv_list(tensor):
    """Transforms a torch.Tensor to a list of multivector objects."""

    tensor = tensor.reshape((-1, 16))
    mv_list = [tensor_to_mv(x) for x in tensor]

    return mv_list


def mv_list_to_tensor(multivectors, batch_shape=None):
    """Transforms a list of multivector objects to a torch.Tensor."""

    tensor = torch.from_numpy(np.array([mv.value for mv in multivectors])).to(torch.float32)
    if batch_shape is not None:
        tensor = tensor.reshape(*batch_shape, 16)

    return tensor


def sample_pin_multivector(spin: bool = False, rng: Optional[np.random.Generator] = None):
    """Samples from the Pin(1,3) group as a product of reflections."""

    if rng is None:
        rng = np.random.default_rng()

    # Sample number of reflections we want to multiply
    if spin:
        i = np.random.randint(3) * 2
    else:
        i = np.random.randint(5)

    # If no reflections, just return unit scalar
    if i == 0:
        return BLADES[""]

    multivector = 1.0
    for _ in range(i):
        # Sample reflection vector
        vector = np.zeros(16)
        
        #vector[1:5] = rng.normal(size=4)
        
        vector[2:5] = rng.normal(size=3) * 2
        norm = np.linalg.norm(vector[2:5])
        vector[1] = ( rng.uniform(size=1) - .5 ) * norm
        
        #norm = rng.normal(size=1) + 5.
        #vector[2:5] = rng.normal(size=3) * 2
        #vector[1] = (norm**2 + np.sum(vector[2:5]**2))**.5
        
        vector_mv = np_to_mv(vector)
        
        vector_mv = vector_mv / abs(vector_mv.mag2()) ** 0.5

        # Multiply together (geometric product)
        multivector = multivector * vector_mv
    '''
    mv_generator_vec = np.zeros(16)
    mv_generator_vec[5:11] = rng.normal(size=6)
    multivector = LAYOUT.MultiVector(mv_generator_vec).exp()
    print("SAMPLED: ", multivector.mag2(), multivector)
    '''
    
    return multivector


def get_parity(mv):
    """Gets parity of a clifford multivector.

    Given a clifford multivector, returns True if it is pure-odd-grade, False if it is pure-even
    grade, and raises a RuntimeError if it is mixed.
    """
    if mv == mv.even:
        return False
    if mv == mv.odd:
        return True
    raise RuntimeError(f"Mixed-grade multivector: {mv}")


def sandwich(u, x):
    """Given clifford multivectors, computes their sandwich product.

    Specifically, given a Pin element u and a PGA element x, both given as clifford multivectors,
    computes the sandwich product
    ```
    sandwich(x, u) = (-1)^(grade(u) * grade(x)) u x u^{-1} .
    ```

    If `u` is of odd grades, then this is equal to `u * grade_involute(x) * u^{-1}`.
    If `u` is of even grades, then this is equal to `u * x * u^{-1}`.
    """

    if get_parity(u):
        return u * x.gradeInvol() * u.shirokov_inverse()

    return u * x * u.shirokov_inverse()


class SlowRandomPinTransform:
    """Random Pin transform on a multivector torch.Tensor.

    Slow, only used for testing purposes. Breaks computational graph.
    """

    def __init__(self, spin=False, rng=None):
        super().__init__()
        self._u = sample_pin_multivector(spin, rng)
        self._u_inverse = self._u.shirokov_inverse()
        '''
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng
        '''

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply Pin transformation to multivector inputs."""
        # Input shape
        assert inputs.shape[-1] == 16
        batch_dims = inputs.shape[:-1]

        # Convert inputs to list of multivectors
        inputs_mv = tensor_to_mv_list(inputs)

        # Transform
        outputs_mv = [sandwich(self._u, x) for x in inputs_mv]
        '''
        outputs_mv = []
        for input_mv in inputs_mv:
            eta = self.rng.normal(size=1) # rapidity
            n = self.rng.normal(size=3)
            n /= np.linalg.norm(n) # direction
            output_mv = clifford.MultiVector(LAYOUT, value=np.zeros(16))
            output_mv[1] = np.cosh(eta) * (input_mv[1] - np.tanh(eta) * np.dot(n, input_mv[2:5]) )
            output_mv[2:5] = input_mv[2:5] + (np.cosh(eta) -1) * np.dot(n, input_mv[2:5]) * n - input_mv[1] * np.sinh(eta) * n
            outputs_mv.append(output_mv)
        '''
        # Back to tensor
        outputs = mv_list_to_tensor(outputs_mv, batch_shape=batch_dims)

        return outputs

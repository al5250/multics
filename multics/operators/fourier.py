from typing import Tuple

import torch
from torch import Tensor
from torch.fft import fftn, ifftn
from scipy.linalg import dft
import numpy as np

from multics.operators.linop import LinearOperator


class Fourier1D(LinearOperator):
    """The fast Fourier transform (FFT)."""

    def __init__(self, length: int) -> None:
        super().__init__()
        self.length = length

    def apply(self, inp: Tensor) -> Tensor:
        out = fftn(inp, dim=-1, norm="ortho")
        return out

    def transpose(self, out: Tensor) -> Tensor:
        inp = ifftn(out, dim=-1, norm="ortho")
        return inp

    @property
    def inp_dim(self) -> int:
        return self.length

    @property
    def out_dim(self) -> int:
        return self.length

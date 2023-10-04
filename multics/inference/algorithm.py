from abc import ABC, abstractmethod
from typing import Tuple, Callable, Optional, Dict

from torch import Tensor
import torch

from multics.operators import LinearOperator


class InferenceAlgorithm(ABC):
    @abstractmethod
    def precompute(
        self, y: Tensor, omega: Tensor, phi: LinearOperator, beta: float
    ) -> Dict[str, Tensor]:
        pass

    @abstractmethod
    def estep(
        self,
        alpha: Tensor,  # K x T x D
        y: Tensor,  # T x D
        omega: Tensor,  # T x D
        phi: LinearOperator,  # D x D
        beta: float,
        **precomputed
    ) -> Tuple[Tensor, Tensor, Tensor]:
        pass

from typing import Optional, Tuple, Callable, Dict
from functools import reduce
from operator import mul

from torch import Tensor
import torch
import numpy as np
from scipy.linalg import inv

from multics.operators import LinearOperator
from multics.inference.algorithm import InferenceAlgorithm


class ExpMax(InferenceAlgorithm):
    """Expectation-Maximization (EM) algorithm for SBL inference.

    The EM algorithm alternates between (1) an E-Step that computes
    posterior statistics (mu, sigma) given current alpha estimates and
    (2) an M-Step that optimizes alpha given (mu, sigma).

    """

    def __init__(self) -> None:
        pass

    def precompute(
        self, y: Tensor, omega: Tensor, phi: LinearOperator, beta: float
    ) -> Dict[str, Tensor]:
        phi_mat = phi.values
        beta_phiT_omega_phi = (
            beta * phi_mat.conj().T @ (omega.unsqueeze(dim=-1) * phi_mat)
        )
        beta_phiT_omega_y = beta * (
            phi_mat.conj().T @ (omega * y).unsqueeze(dim=-1)
        ).squeeze(dim=-1)
        return {
            "beta_phiT_omega_phi": beta_phiT_omega_phi,
            "beta_phiT_omega_y": beta_phiT_omega_y,
        }

    def estep(
        self,
        alpha: Tensor,  # K x T x D
        y: Tensor,  # T x D
        omega: Tensor,  # T x D
        phi: LinearOperator,  # D x D
        beta: float,
        beta_phiT_omega_phi: Tensor,  # T x D x D
        beta_phiT_omega_y: Tensor,  # T x D
    ) -> Tuple[Tensor, Tensor, Tensor]:
        precision = beta_phiT_omega_phi.expand(
            (len(alpha), -1, -1, -1)
        ).clone()  # K x T x D x D
        diag_idx = torch.arange(precision.size(dim=-1))
        precision[..., diag_idx, diag_idx] += alpha
        sigma = torch.inverse(precision, out=precision)  # K x T x D x D
        mu = (sigma @ beta_phiT_omega_y.unsqueeze(dim=-1)).squeeze(dim=-1)  # K x T x D
        sigma_diag = torch.diagonal(sigma, dim1=-2, dim2=-1)  # K x T x D
        log_like = self._log_likelihood(y, omega, phi, alpha, beta, mu, sigma)  # K x T

        del precision
        return mu, sigma_diag, log_like

    def _log_likelihood(
        self,
        y: Tensor,  # T x D
        omega: Tensor,  # T x D
        phi: LinearOperator,  # D x D
        alpha: Tensor,  # K x T x D
        beta: float,
        mu: Tensor,  # K x T x D
        sigma: Tensor,  # K x T x D x D
    ) -> Tensor:
        recon_loss = (beta / 2) * torch.norm(y - omega * phi(mu), dim=-1) ** 2
        reg_loss = (1 / 2) * torch.sum(alpha * (mu.conj() * mu), dim=-1)
        norm_constant = (1 / 2) * (-torch.logdet(sigma) - torch.log(alpha).sum(dim=-1))
        assert recon_loss.size() == reg_loss.size() == norm_constant.size()
        log_like = -(recon_loss + reg_loss + norm_constant)
        if torch.is_complex(log_like):
            # assert torch.all(torch.abs(log_like.imag) < 1e-6)
            log_like = log_like.real
        # print("log_like", log_like)
        return log_like

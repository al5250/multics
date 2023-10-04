from typing import Optional, Callable

import torch
from torch import Tensor

from multics.inference import ExpMax, CovFreeExpMax
from multics.operators import LinearOperator


class MultiTaskCompSens:
    def __init__(
        self, mode: str, alg: str, num_clusters: Optional[int] = None, **kwargs
    ):

        assert mode in ("separate", "joint", "clustered")
        if mode == "clustered":
            assert isinstance(num_clusters, int)
            self.num_clusters = num_clusters
        self.mode = mode

        if alg == "em":
            self.alg = ExpMax(**kwargs)
        elif alg == "cofem":
            self.alg = CovFreeExpMax(**kwargs)
        else:
            raise ValueError("Valid algorithm options are either 'em' or 'cofem'.")

    def fit(
        self,
        data: Tensor,
        sensor: LinearOperator,
        noise_std: float,
        num_iters: int,
        masks: Optional[Tensor] = None,
        logger: Optional[Callable] = None,
    ):

        if masks is None:
            masks = torch.ones_like(data, dtype=torch.bool)

        beta = 1 / (noise_std**2)
        precomputed = self.alg.precompute(data, masks, sensor, beta)

        if self.mode == "separate":
            # Different alpha vector for every data vector
            alpha = torch.ones((1, len(data), sensor.inp_dim), device=data.device)

            mu, sigma_diag, _ = self.alg.estep(
                alpha, data, masks, sensor, beta, **precomputed
            )
            if logger is not None:
                logger(0, mu[0])

            for t in range(num_iters):
                # M-Step
                alpha_new = 1 / ((mu.conj() * mu) + sigma_diag)
                if torch.is_complex(alpha_new):
                    # assert torch.all(torch.abs(alpha_new.imag) < 1e-6)
                    alpha_new = alpha_new.real

                # E-Step
                alpha = alpha_new
                mu, sigma_diag, _ = self.alg.estep(
                    alpha, data, masks, sensor, beta, **precomputed
                )
                if logger is not None:
                    logger(t + 1, mu[0])

            self.alpha = alpha
            self.mu = mu[0]
            self.sigma_diag = sigma_diag[0]

        elif self.mode == "joint":
            # Same alpha vector for every data vector
            alpha = torch.ones((1, 1, sensor.inp_dim), device=data.device).expand(
                (1, len(data), sensor.inp_dim)
            )

            mu, sigma_diag, _ = self.alg.estep(
                alpha, data, masks, sensor, beta, **precomputed
            )
            if logger is not None:
                logger(0, mu[0])

            for t in range(num_iters):
                # M-Step
                alpha_new = 1 / ((mu.conj() * mu) + sigma_diag).mean(
                    dim=1, keepdim=True
                )
                if torch.is_complex(alpha_new):
                    # assert torch.all(torch.abs(alpha_new.imag) < 1e-6)
                    alpha_new = alpha_new.real

                # E-Step
                alpha = alpha_new.expand((1, len(data), sensor.inp_dim))
                mu, sigma_diag, _ = self.alg.estep(
                    alpha, data, masks, sensor, beta, **precomputed
                )
                if logger is not None:
                    logger(t + 1, mu[0])

            self.alpha = alpha
            self.mu = mu[0]
            self.sigma_diag = sigma_diag[0]

        else:
            # Mixture of alpha vectors
            alpha = torch.ones(
                (self.num_clusters, 1, sensor.inp_dim), device=data.device
            ).expand((self.num_clusters, len(data), sensor.inp_dim))

            alpha = alpha + 1e-5 * torch.randn_like(alpha)

            mu, sigma_diag, log_like = self.alg.estep(
                alpha, data, masks, sensor, beta, **precomputed
            )
            probs = torch.softmax(log_like, dim=0)
            idx = torch.arange(len(data), device=mu.device)
            assignment = probs.argmax(dim=0)
            if logger is not None:
                logger(0, mu[assignment, idx])

            for t in range(num_iters):

                # M-Step
                probs = probs.unsqueeze(dim=-1)
                alpha_new = probs.sum(dim=1, keepdim=True) / (
                    probs * ((mu.conj() * mu) + sigma_diag)
                ).sum(dim=1, keepdim=True)
                if torch.is_complex(alpha_new):
                    # assert torch.all(torch.abs(alpha_new.imag) < 1e-6)
                    alpha_new = alpha_new.real

                # E-Step
                alpha = alpha_new.expand((self.num_clusters, len(data), sensor.inp_dim))
                mu, sigma_diag, log_like = self.alg.estep(
                    alpha, data, masks, sensor, beta, **precomputed
                )
                probs = torch.softmax(log_like, dim=0)
                assignment = probs.argmax(dim=0)

                if logger is not None:
                    logger(t + 1, mu[assignment, idx])

                # print("probs", probs)

            self.alpha = alpha
            self.mu = mu[assignment, idx]
            self.sigma_diag = sigma_diag[assignment, idx]

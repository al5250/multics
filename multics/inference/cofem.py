from typing import Tuple, Optional, Callable, Union, List, Dict
import warnings

import torch
from torch.distributions import Bernoulli
from torch import Tensor
import numpy as np

from multics.operators import LinearOperator
from multics.inference.algorithm import InferenceAlgorithm
from multics.inference.em import ExpMax


class CovFreeExpMax(InferenceAlgorithm):

    """Time/space-efficient Expectation-Maximization algorithm powered by
    conjugate gradient."""

    def __init__(
        self,
        num_probes: int = 10,
        cg_tol: float = 1e-5,
        max_cg_iters: int = 1000,
        precondition: bool = False,
    ) -> None:
        self.num_probes = num_probes
        self.cg_tol = cg_tol
        self.max_cg_iters = max_cg_iters
        self.precondition = precondition

    def precompute(
        self, y: Tensor, omega: Tensor, phi: LinearOperator, beta: float
    ) -> Dict[str, Tensor]:
        phiT_omega_y = phi.T(omega * y)
        return {"phiT_omega_y": phiT_omega_y}

    def estep(
        self,
        alpha: Tensor,  # K x T x D
        y: Tensor,  # T x D
        omega: Tensor,  # T x D
        phi: LinearOperator,  # D x D
        beta: float,
        phiT_omega_y: Tensor,  # T x D
    ) -> Tuple[Tensor, Tensor, Optional[int], Optional[Tensor], Optional[Tensor]]:
        phiT_omega_y = phiT_omega_y.unsqueeze(dim=0).expand(
            (len(alpha), -1, -1)  # K x T x D
        )
        probes = self._samp_probes(
            size=(self.num_probes,) + phiT_omega_y.shape,
            device=phiT_omega_y.device,
        )  # Q x K x T x D

        if self.precondition:
            M_inv = lambda x: x / (beta + alpha)
            rotated_probes = torch.sqrt(beta + alpha) * probes
            b = torch.cat([beta * phiT_omega_y.unsqueeze(dim=0), rotated_probes], dim=0)
        else:
            M_inv = lambda x: x
            b = torch.cat([beta * phiT_omega_y.unsqueeze(dim=0), probes], dim=0)

        A = lambda x: beta * phi.T(omega * phi(x)) + alpha * x

        # phi_mat = phi.values
        # beta_phiT_omega_phi = (
        #     beta * phi_mat.conj().T @ (omega.unsqueeze(dim=-1) * phi_mat)
        # )
        # mat = beta_phiT_omega_phi + torch.diag_embed(alpha)
        # Minv = 1 / (alpha + beta)
        # mat = Minv.unsqueeze(dim=-1) * mat
        # print("True", (torch.logdet(mat) - torch.log(alpha).sum(dim=-1)) / 2)

        x, converge_iter, coeffs1, coeffs2 = self._conj_grad(
            A,
            b,
            dim=-1,
            M_inv=M_inv,
            max_iters=self.max_cg_iters,
            tol=self.cg_tol,
        )  # (Q + 1) x K x T x D
        print("converge iter", converge_iter)
        coeffs1 = coeffs1[1:]  # Q x K x T x U
        coeffs2 = coeffs2[1:]  # Q x K x T x U

        mu = x[0]  # K x T x D
        out = x[1:]  # Q x K x T x D

        if self.precondition:
            out = out / torch.sqrt(beta + alpha)

        if torch.is_complex(out):
            out = out.real

        sigma_diag = (probes * out).mean(dim=0).clamp(min=0)

        log_like = self._log_likelihood(
            y, omega, phi, alpha, beta, mu, coeffs1, coeffs2
        )
        return mu, sigma_diag, log_like

    def _log_likelihood(
        self,
        y: Tensor,  # T x D
        omega: Tensor,  # T x D
        phi: LinearOperator,  # D x D
        alpha: Tensor,  # K x T x D
        beta: float,
        mu: Tensor,  # K x T x D
        coeffs1: Tensor,  # Q x K x T x U
        coeffs2: Tensor,  # Q x K x T x U
    ) -> Tensor:
        main_diag = 1 / coeffs1
        main_diag[..., 1:] += coeffs2 / coeffs1[..., :-1]
        off_diag = -torch.sqrt(coeffs2) / coeffs1[..., :-1]
        tridiag = torch.diag_embed(main_diag) + torch.diag_embed(off_diag, offset=-1)
        evals, evecs = torch.linalg.eigh(tridiag, UPLO="L")
        evecs1 = evecs[..., 0, :]
        sigma_inv_logdets = phi.inp_dim * (
            torch.sum(torch.log(evals) * (evecs1.conj() * evecs1), dim=-1)  # Q x K x T
        )
        if self.precondition:
            norm_constant = (1 / 2) * (
                torch.mean(sigma_inv_logdets, dim=0)
                + torch.log(1 + beta / alpha).sum(dim=-1)
            )
        else:
            norm_constant = (1 / 2) * (
                torch.mean(sigma_inv_logdets, dim=0) - torch.log(alpha).sum(dim=-1)
            )
        if torch.isnan(norm_constant).any():
            import pdb

            pdb.set_trace()
        # print(
        #     "estimate",
        #     torch.mean(sigma_inv_logdets, dim=0) + torch.log(alpha + beta).sum(dim=-1),
        # )
        recon_loss = (beta / 2) * torch.norm(y - omega * phi(mu), dim=-1) ** 2
        reg_loss = (1 / 2) * torch.sum(alpha * (mu.conj() * mu), dim=-1)
        assert recon_loss.size() == reg_loss.size() == norm_constant.size()
        log_like = -(recon_loss + reg_loss + norm_constant)
        if torch.is_complex(log_like):
            # assert torch.all(torch.abs(log_like.imag) < 1e-6)
            log_like = log_like.real
        # print("log like", log_like)
        return log_like

    @staticmethod
    def _samp_probes(size: Tuple[int, ...], device: str = "cpu") -> Tensor:
        p = torch.tensor(0.5, device=device)
        z = 2 * Bernoulli(probs=p).sample(size) - 1
        return z

    @staticmethod
    def _conj_grad(
        A: Callable[[Tensor], Tensor],
        b: Tensor,
        dim: Union[int, Tuple[int, ...]],
        M_inv: Optional[Callable[[Tensor], Tensor]] = None,
        max_iters: int = 1000,
        tol: float = 1e-10,
    ) -> Tuple[Tensor, Optional[int], Optional[Tensor], Optional[Tensor]]:
        x = torch.zeros_like(b)
        r = b
        z = r if M_inv is None else M_inv(r)
        p = z
        rz = torch.sum(r.conj() * z, dim=dim, keepdim=True)
        # z2 = torch.sum(z.conj() * z, dim=dim, keepdim=True)
        t = 0
        alphas = []
        betas = []

        for t in range(max_iters):
            Ap = A(p)
            pAp = torch.sum(p.conj() * Ap, dim=dim, keepdim=True)
            alpha = rz / pAp
            alphas.append(alpha.squeeze(dim=dim))

            x = x + alpha * p
            r = r - alpha * Ap

            eps = (torch.norm(r) / torch.norm(b)) ** 2
            if eps < tol:
                coeffs1 = torch.stack(alphas, dim=-1)
                coeffs2 = torch.stack(betas, dim=-1)
                return x, t, coeffs1, coeffs2

            rz_old = rz
            # z2_old = z2
            z = r if M_inv is None else M_inv(r)
            # z2 = torch.sum(z.conj() * z, dim=dim, keepdim=True)
            # beta = z2 / z2_old
            rz = torch.sum(r.conj() * z, dim=dim, keepdim=True)
            beta = rz / rz_old
            if t < max_iters - 1:
                betas.append(beta.squeeze(dim=dim))
            p = z + beta * p

        warnings.warn(
            f"Conjugate gradient failed to converge to a tolerance level of {tol:.3e} "
            f"after {max_iters} iterations.  Exiting with an error of {eps:.3e}.",
            stacklevel=1,
        )
        coeffs1 = torch.stack(alphas, dim=-1)
        coeffs2 = torch.stack(betas, dim=-1)
        return x, None, coeffs1, coeffs2

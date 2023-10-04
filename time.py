import torch
import numpy as np
import argparse
import psutil
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from memory_profiler import memory_usage


from multics.model import MultiTaskCompSens
from multics.operators import DenseMatrix, DiscreteCosine1D, Fourier1D


parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--dim", dest="dim", type=int, required=True)
parser.add_argument("--alg", dest="alg", type=str, required=True)
parser.add_argument("--device", dest="device", type=str, required=True)
parser.add_argument("--num", dest="num", type=int, required=True)
parser.add_argument("--two_clust", action="store_true")
parser.add_argument("--no_two_clust", dest="two_clust", action="store_false")
parser.set_defaults(two_clust=True)
parser.add_argument("--clusts", dest="clusts", type=int, required=False)

args = parser.parse_args()

torch.set_default_device(args.device)
torch.set_default_dtype(torch.float)

seed = 8888
torch.manual_seed(seed)
noise_std = 0.05

if args.two_clust:
    dic = Fourier1D(length=args.dim)
    idx = torch.stack(
        [torch.randperm(args.dim)[: int(0.75 * args.dim)] for i in range(args.num)]
    )
    sigs = torch.randn((args.num, args.dim))
    sigs[: args.num // 2, int(0.05 * args.dim) :] = 0
    sigs[args.num // 2 :, : -int(0.05 * args.dim)] = 0
    data = dic(sigs)
    noise = noise_std * torch.randn_like(data)
    mask = torch.ones_like(sigs, dtype=torch.bool)
    for m, i in zip(mask, idx):
        m[i] = 0.0
    data[~mask] = 0.0
    num_clusters = 2
else:
    print("Multi-cluster")
    dic = Fourier1D(length=args.dim)
    idx = torch.stack(
        [torch.randperm(args.dim)[: int(0.75 * args.dim)] for i in range(args.num)]
    )
    sigs = torch.randn((args.num, args.dim))
    num_clusters = args.clusts
    num_sparse = int(0.05 * args.dim)

    counter = 0
    for i in range(args.num):
        assign = counter % num_clusters
        sigs[i, : (num_sparse * assign)] = 0

        sigs[i, (num_sparse * (assign + 1)) :] = 0
        counter += 1

    assert torch.all(torch.sum(sigs != 0, dim=-1) == num_sparse)
    data = dic(sigs)
    noise = noise_std * torch.randn_like(data)
    mask = torch.ones_like(sigs, dtype=torch.bool)
    for m, i in zip(mask, idx):
        m[i] = 0.0
    data[~mask] = 0.0


class CorrectedSummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)


writer = CorrectedSummaryWriter()
writer.add_hparams(
    {"dim": args.dim, "alg": args.alg, "device": args.device}, {"metric": 0}
)


def log(t, mu, sigs):
    writer.add_scalar(
        "Error/err", torch.norm(sigs - mu) / torch.norm(sigs), global_step=t
    )
    if args.device == "cpu":
        pid = os.getpid()
        python_process = psutil.Process(pid)
        memoryUse = python_process.memory_info()[0] / 2.0**30
        writer.add_scalar("Error/mem", memoryUse, global_step=t)
    elif args.device == "cuda":
        writer.add_scalar(
            "Error/mem", torch.cuda.max_memory_allocated(0) / 1e9, global_step=t
        )


if args.alg == "em":
    model = MultiTaskCompSens(mode="clustered", alg="em", num_clusters=num_clusters)
elif args.alg == "cofem":
    model = MultiTaskCompSens(
        mode="clustered",
        alg="cofem",
        num_clusters=num_clusters,
        num_probes=15,
        cg_tol=1e-10,
        precondition=True,
    )
else:
    raise ValueError()


def f():
    model.fit(
        data,
        dic,
        noise_std,
        num_iters=50,
        masks=mask,
        logger=lambda t, mu: log(t, mu, sigs),
    )


mem_usage = memory_usage(f)
print("Max mem usage", max(mem_usage) / 1e3)

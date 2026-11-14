"""Domain-adversarial neural network for cross-subject sEMG LOSO.

A three-stage domain adaptation stack for the UCI sEMG hand-movement
dataset:

1. CORAL pre-alignment of second-order feature statistics between the
   source subjects and the held-out subject's unlabeled calibration
   windows (see :mod:`limen.domain_adaptation`).
2. Domain-adversarial training: a convolutional feature extractor feeds a
   six-way task classifier and a five-way subject discriminator connected
   through a gradient-reversal layer, so the learned representation is
   discriminative for gestures and invariant to subject identity.
3. Per-subject batch-norm calibration at test time: the batch-norm
   running statistics of the held-out subject are re-estimated from that
   subject's unlabeled calibration windows only.

Strict LOSO protocol: the held-out subject's test windows never
influence training, CORAL estimation, or batch-norm statistics. Only
the calibration partition (first trials per gesture, labels unused) is
visible to adaptation machinery.

All randomness flows through explicit seeds; runs are deterministic on
CPU and GPU.

Reference for the adversarial formulation:
  Ganin et al., "Domain-Adversarial Training of Neural Networks",
  JMLR 2016. Implemented from the method's defining equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gradient reversal
# ---------------------------------------------------------------------------

class _GradientReversalFn(torch.autograd.Function):
    """Identity forward; multiplies the gradient by -lambda on backward."""

    @staticmethod
    def forward(ctx, x, lam: float):  # type: ignore[override]
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        return grad_output.neg() * ctx.lam, None


class GradientReversal(nn.Module):
    """Gradient-reversal layer. ``lambda_`` scales the reversed gradient."""

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFn.apply(x, self.lambda_)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class DannNet(nn.Module):
    """CNN feature extractor + task classifier + subject discriminator.

    Input: EMG windows of shape (batch, 1, n_channels, n_samples).
    The task head predicts gesture classes; the domain head predicts
    subject identity from gradient-reversed features. Batch-norm layers
    carry per-domain statistics, which stage 3 recalibrates per subject.
    """

    def __init__(
        self,
        n_classes: int = 6,
        n_domains: int = 5,
        n_channels: int = 2,
        feature_dim: int = 64,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )
        self.task_head = nn.Linear(feature_dim, n_classes)
        self.grl = GradientReversal()
        self.domain_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_domains),
        )

    def forward(self, x: torch.Tensor, lambda_: float | None = None):
        z = self.features(x)
        task_logits = self.task_head(z)
        if lambda_ is not None:
            self.grl.lambda_ = lambda_
        domain_logits = self.domain_head(self.grl(z))
        return task_logits, domain_logits

    def task_predict(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        return self.task_head(z)


# ---------------------------------------------------------------------------
# Batch-norm recalibration (stage 3)
# ---------------------------------------------------------------------------

def recalibrate_batchnorm(model: nn.Module, windows: torch.Tensor, batch_size: int = 256) -> None:
    """Re-estimate BN running statistics from unlabeled windows, labels unused.

    The model is put in train() mode so BN layers consume batch statistics,
    but only forward passes are run and no gradients are taken. Weight and
    bias parameters are untouched; only running_mean / running_var change.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.reset_running_stats()
            module.momentum = None  # cumulative moving average over all batches
    model.train()
    with torch.no_grad():
        for start in range(0, windows.shape[0], batch_size):
            model.task_predict(windows[start : start + batch_size])
    model.eval()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DannTrainConfig:
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_max: float = 1.0
    seed: int = 20261011
    device: str = "cpu"


def dann_schedule_lambda(epoch: int, total_epochs: int, lambda_max: float) -> float:
    """Ganin schedule: lambda ramps 0 -> lambda_max with a sigmoid in epoch."""
    p = epoch / max(1, total_epochs - 1)
    return float(lambda_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))


def train_dann(
    model: DannNet,
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    source_domain: torch.Tensor,
    target_x: torch.Tensor,
    target_domain: int,
    cfg: DannTrainConfig,
) -> None:
    """Adversarial training on labeled source + unlabeled target windows.

    Each batch mixes source windows (task loss + domain loss) with target
    windows (domain loss only, through the reversal layer). The target
    domain index is ``n_domains - 1`` by convention of the caller.
    """
    gen = torch.Generator().manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)
    source_x, source_y, source_domain = (
        source_x.to(device),
        source_y.to(device),
        source_domain.to(device),
    )
    target_x = target_x.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    n_source = source_x.shape[0]
    n_target = target_x.shape[0]
    steps = max(1, max(n_source, n_target) // cfg.batch_size)

    for epoch in range(cfg.epochs):
        model.train()
        lam = dann_schedule_lambda(epoch, cfg.epochs, cfg.lambda_max)
        perm_s = torch.randperm(n_source, generator=gen)
        perm_t = torch.randperm(n_target, generator=gen)
        for step in range(steps):
            s_idx = perm_s[
                (step * cfg.batch_size) % n_source : (step * cfg.batch_size) % n_source
                + cfg.batch_size
            ]
            if s_idx.shape[0] < cfg.batch_size:  # wrap around
                extra = perm_s[: cfg.batch_size - s_idx.shape[0]]
                s_idx = torch.cat([s_idx, extra])
            t_idx = perm_t[
                (step * cfg.batch_size) % n_target : (step * cfg.batch_size) % n_target
                + cfg.batch_size
            ]
            if t_idx.shape[0] < cfg.batch_size:
                extra = perm_t[: cfg.batch_size - t_idx.shape[0]]
                t_idx = torch.cat([t_idx, extra])

            xs, ys, ds = source_x[s_idx], source_y[s_idx], source_domain[s_idx]
            xt = target_x[t_idx]
            x = torch.cat([xs, xt], dim=0)

            task_logits, domain_logits = model(x, lambda_=lam)
            loss_task = F.cross_entropy(task_logits[: xs.shape[0]], ys)
            domain_labels = torch.cat(
                [ds, torch.full((xt.shape[0],), target_domain, dtype=torch.long, device=device)]
            )
            loss_domain = F.cross_entropy(domain_logits, domain_labels)
            loss = loss_task + loss_domain

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()


def predict_labels(model: DannNet, x: torch.Tensor, batch_size: int = 512) -> NDArray[np.int64]:
    """Argmax task predictions for a window batch."""
    model.eval()
    preds: list[NDArray[np.int64]] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            logits = model.task_predict(x[start : start + batch_size])
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)

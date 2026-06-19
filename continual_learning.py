"""
continual_learning.py
=====================
Continual learning components for CADRE:

  OnlineEWC  — self-scaling, similarity-aware online EWC (CADRE core, M1+M2+M3)
  EWC        — vanilla fixed-λ offline EWC (ablation baseline)
  EMA        — Polyak weight averaging applied at evaluation time (SWA-style)
  precompute_frozen_anchor() — cache g_frozen(x) for the anchor probe set
  anchor_loss()              — L_anchor = ||g_LoRA(x) - g_frozen(x)||²  [Eq. 4]

Theory correspondence
---------------------
  M1  sum-normalised Fisher      → bounded consolidation mass  (Proposition 1)
  M2  self-scaling λ_t           → scale-invariant trade-off   (Proposition 2)
  M3  similarity-aware γ_sim     → Eq. 3  (heuristic, validated empirically)
  L_anchor                       → Eq. 4  (soft bound on expected drift)

References
----------
  Online EWC:  Schwarz et al., Progress & Compress, ICML 2018
  EWC:         Kirkpatrick et al., PNAS 2017
  LwF / proxy: Li & Hoiem, TPAMI 2017
  SWA / EMA:   Izmailov et al., UAI 2018
"""

from __future__ import annotations

import copy
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CFG, AMP


# =============================================================================
# HELPERS
# =============================================================================

def _autocast():
    """Return the appropriate autocast context for the current device."""
    if AMP:
        return torch.cuda.amp.autocast()
    import contextlib
    return contextlib.nullcontext()


def _named_trainable(model: nn.Module) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield (name, parameter) pairs for all trainable parameters."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield name, param


# =============================================================================
# 1. ANCHOR UTILITIES
# =============================================================================

def precompute_frozen_anchor(
    model: nn.Module,
    anchor_x: torch.Tensor,
) -> torch.Tensor:
    """
    Cache frozen-prior embeddings g_frozen(x) for the anchor probe set A.

    LoRA adapters are temporarily disabled so the encoder runs with its
    original pretrained weights (LoRA delta = 0), giving the true prior
    embedding that L_anchor is anchored to.

    The probe set A is fixed once at the start of the first modality and
    never updated — its embeddings are cached here and reused for all
    subsequent modalities, bounding drift from the original prior rather
    than from the previous task's representation (cf. LwF which distils
    from the previous model state).

    Parameters
    ----------
    model    : CADREModel — the model whose LoRA adapters will be toggled
    anchor_x : torch.Tensor  shape (|A|, 3, H, W) — probe images on device

    Returns
    -------
    torch.Tensor  shape (|A|, embed_dim) — detached frozen embeddings
    """
    from model import set_lora_enabled  # local import avoids circular dependency

    set_lora_enabled(model.lora, False)   # bypass LoRA → pure frozen backbone
    model.eval()

    with torch.no_grad(), _autocast():
        feats = model.vision_enc(anchor_x)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]

    set_lora_enabled(model.lora, True)    # restore LoRA
    model.train()

    return feats.detach()


def anchor_loss(
    model: nn.Module,
    anchor_x: torch.Tensor,
    z_frozen: torch.Tensor,
) -> torch.Tensor:
    """
    Anchor-to-prior regularisation penalty  (Eq. 4 in paper).

        L_anchor = (1/|A|) Σ_{x ∈ A} ||g_LoRA(x) - g_frozen(x)||²

    Minimising β · L_anchor is the Lagrangian of:
        min_θ L_CE   subject to  mean drift on A ≤ δ
    providing a soft bound on expected embedding drift from the frozen
    prior — limiting shortcut adoption without a per-input Lipschitz
    guarantee.

    Parameters
    ----------
    model    : CADREModel — current adapted model (LoRA enabled)
    anchor_x : torch.Tensor  shape (|A|, 3, H, W) — probe images on device
    z_frozen : torch.Tensor  shape (|A|, embed_dim) — cached frozen embeddings

    Returns
    -------
    torch.Tensor  scalar — mean squared drift
    """
    feats = model.vision_enc(anchor_x)
    if isinstance(feats, (list, tuple)):
        feats = feats[0]
    return F.mse_loss(feats, z_frozen)


# =============================================================================
# 2. ONLINE EWC  (CADRE core: M1 + M2 + M3)
# =============================================================================

class OnlineEWC:
    """
    CADRE's redesigned online EWC with three scale/order fixes.

    Standard online EWC (Schwarz et al., ICML 2018) keeps a running Fisher
    and reference point:
        L_EWC(θ) = Σ_i F_i (θ_i − θ*_i)²
        F ← γ · F + F̂^(t)

    This has two coupled scale problems (§2 / §3 of paper):
      (a) Raw Fisher accumulates unboundedly → order-dependent penalty magnitude
      (b) A fixed global λ is not comparable across tasks/steps

    CADRE removes both by construction:

    M1 — Sum-normalised per-task Fisher
         F̂^(t) ← F̂^(t) / Σ_i F̂^(t)_i
         Every modality contributes the same total importance (mass = 1).
         → Proposition 1: total mass m_t ≤ 1/(1−γ) for all t  (bounded)

    M2 — Self-scaling strength  (per step, from detached loss values)
         λ_t = min( ρ · L_CE / (L_EWC + ε),  λ_max )
         The EWC penalty contributes a fixed fraction ρ of the task loss.
         → Proposition 2: value λ_t · L_EWC = ρ · L_CE and gradient
           (λ_t/c)∇(c·L_EWC) = λ_t∇L_EWC are both scale-invariant.
         (Note: M2 is applied in trainer.py, not here; λ_t is computed
          from the live loss values during the training loop.)

    M3 — Similarity-aware retention
         γ_sim = γ · (0.5 + 0.5 · cos(p_t, p_{t−1}))     [Eq. 3]
         Dissimilar modalities relax retention (lower γ_sim) so the
         penalty doesn't over-constrain incompatible representations.
         Satisfies γ_sim ∈ [0, γ], the precondition of Proposition 1.

    Parameters
    ----------
    gamma    : float — base Fisher decay factor γ (default CFG["ewc_gamma"])
    use_sim  : bool  — whether to apply M3 similarity-aware γ_sim modulation
    """

    def __init__(
        self,
        gamma:   float = None,
        use_sim: bool  = True,
    ):
        self.gamma   = gamma if gamma is not None else CFG["ewc_gamma"]
        self.use_sim = use_sim

        # Running sum-normalised Fisher  {param_name: tensor}
        self._F: Dict[str, torch.Tensor] = {}

        # Reference parameters θ* after last consolidation  {name: tensor}
        self._theta_star: Dict[str, torch.Tensor] = {}

        # Prototype of the most recently consolidated modality
        self._last_proto: Optional[torch.Tensor] = None

    # ── Fisher estimation ─────────────────────────────────────────────────────

    def _estimate_fisher(
        self,
        model:     nn.Module,
        loader:    DataLoader,
        n_batches: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate the empirical Fisher diagonal for trainable parameters.

        F̂_i ≈ E[ (∂ log p(y|x) / ∂θ_i)² ]
             ≈ (1/N) Σ_batch (∂ L_CE / ∂θ_i)²

        After estimation, M1 normalisation is applied:
            F̂ ← F̂ / Σ_i F̂_i   (unit total mass)

        Parameters
        ----------
        model     : CADREModel — model after training on the current modality
        loader    : DataLoader — training loader for the current modality
        n_batches : int        — number of batches used for estimation

        Returns
        -------
        dict  {param_name: Fisher diagonal tensor}  (sum-normalised, on device)
        """
        model.eval()
        F_hat: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in _named_trainable(model)
        }
        count = 0

        for x, y, _ in loader:
            if count >= n_batches:
                break
            x = x.to(CFG["device"])
            y = y.to(CFG["device"])

            model.zero_grad()
            with _autocast():
                logits = model(x)
                loss   = F.cross_entropy(logits, y)

            loss.backward()

            for name, param in _named_trainable(model):
                if param.grad is not None:
                    F_hat[name] += param.grad.detach() ** 2

            count += 1

        if count == 0:
            raise RuntimeError("Fisher estimation: loader yielded no batches.")

        # Average over batches
        for name in F_hat:
            F_hat[name] /= count

        # M1: sum-normalise to unit total mass (Proposition 1)
        total_mass = sum(v.sum().item() for v in F_hat.values()) + 1e-12
        for name in F_hat:
            F_hat[name] /= total_mass

        model.train()
        return F_hat

    # ── Modality prototype ────────────────────────────────────────────────────

    def proto(
        self,
        model:  nn.Module,
        loader: DataLoader,
    ) -> torch.Tensor:
        """
        Compute the modality prototype p_t as the mean of L2-normalised
        image embeddings over the training set.

        Used by M3 to compute cosine similarity between consecutive modalities:
            cos(p_t, p_{t-1}) ∈ [-1, 1]
            → γ_sim = γ · (0.5 + 0.5 · cos)  ∈ [0, γ]

        Parameters
        ----------
        model  : CADREModel — model after training on the current modality
        loader : DataLoader — training loader for the current modality

        Returns
        -------
        torch.Tensor  shape (embed_dim,) — unit-normed prototype on CPU
        """
        model.eval()
        vecs = []

        with torch.no_grad():
            for x, _, _ in loader:
                with _autocast():
                    feats = model.vision_enc(x.to(CFG["device"]))
                if isinstance(feats, (list, tuple)):
                    feats = feats[0]
                # L2-normalise each embedding before averaging
                vecs.append(F.normalize(feats.float(), dim=-1).cpu())

        model.train()
        return torch.cat(vecs, dim=0).mean(dim=0)  # shape: (embed_dim,)

    # ── γ_sim computation  (M3) ───────────────────────────────────────────────

    def _gamma_sim(self, new_proto: torch.Tensor) -> float:
        """
        Compute similarity-aware decay factor γ_sim  [Eq. 3].

        If no previous prototype exists (first modality), returns the base γ.

        Parameters
        ----------
        new_proto : torch.Tensor  shape (embed_dim,) — current modality prototype

        Returns
        -------
        float  γ_sim ∈ [0, γ]
        """
        if not self.use_sim or self._last_proto is None:
            return self.gamma

        cos = F.cosine_similarity(
            new_proto.unsqueeze(0),
            self._last_proto.unsqueeze(0),
        ).item()  # scalar ∈ [-1, 1]

        # Eq. 3:  γ_sim = γ · (0.5 + 0.5 · cos)
        # When cos = 1  (identical modalities): γ_sim = γ   (full retention)
        # When cos = -1 (opposite modalities):  γ_sim = 0   (no retention)
        gamma_sim = self.gamma * (0.5 + 0.5 * cos)
        return float(gamma_sim)

    # ── Consolidation ─────────────────────────────────────────────────────────

    def consolidate(
        self,
        model:     nn.Module,
        loader:    DataLoader,
        proto:     torch.Tensor,
        n_batches: int = None,
    ) -> None:
        """
        Consolidate the current modality into the running Fisher and
        update the reference parameters θ*.

        Called once after training on each modality, before the next
        modality's data arrives.

        Update rule (combining M1 + M3):
            F ← γ_sim · F  +  F̂^(t)        [F̂^(t) is already sum-normalised]
            θ* ← θ_t                         [reference = current weights]

        Parameters
        ----------
        model     : CADREModel  — model after training on the current modality
        loader    : DataLoader  — training loader (same modality)
        proto     : torch.Tensor — modality prototype from self.proto()
        n_batches : int, optional — overrides CFG["fisher_batches"] if set
        """
        n_batches = n_batches or CFG["fisher_batches"]

        # Estimate sum-normalised Fisher for current modality (M1)
        F_hat = self._estimate_fisher(model, loader, n_batches)

        # Compute similarity-aware γ_sim (M3)
        gamma_sim = self._gamma_sim(proto)

        if not self._F:
            # First modality: initialise running Fisher directly
            self._F = {name: v.clone() for name, v in F_hat.items()}
        else:
            # Subsequent modalities: decay-and-accumulate with M3 γ_sim
            for name, f_new in F_hat.items():
                if name in self._F:
                    self._F[name] = gamma_sim * self._F[name] + f_new
                else:
                    self._F[name] = f_new

        # Update reference parameters θ* to current weights
        self._theta_star = {
            name: param.detach().clone()
            for name, param in _named_trainable(model)
        }

        # Store prototype for next modality's M3 computation
        self._last_proto = proto.detach().cpu()

        print(
            f"[OnlineEWC] Consolidated | γ_sim={gamma_sim:.4f} | "
            f"F mass={sum(v.sum().item() for v in self._F.values()):.4f}"
        )

    # ── Penalty  (L_EWC) ─────────────────────────────────────────────────────

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the EWC quadratic penalty  L_EWC(θ).

            L_EWC(θ) = Σ_i F_i · (θ_i − θ*_i)²

        Returns zero before the first consolidation (first modality training).

        The self-scaling multiplier λ_t (M2) is applied by the trainer,
        not here, so this method returns the raw unscaled penalty.

        Parameters
        ----------
        model : CADREModel — current model (in training mode)

        Returns
        -------
        torch.Tensor  scalar on CFG["device"]
        """
        if not self._F:
            return torch.tensor(0.0, device=CFG["device"])

        loss = torch.tensor(0.0, device=CFG["device"])
        for name, param in _named_trainable(model):
            if name in self._F and name in self._theta_star:
                diff    = param - self._theta_star[name]
                loss    = loss + (self._F[name] * diff ** 2).sum()

        return loss


# =============================================================================
# 3. VANILLA EWC  (fixed-λ offline ablation baseline)
# =============================================================================

class EWC:
    """
    Vanilla fixed-λ offline EWC (Kirkpatrick et al., PNAS 2017).

    Uses a fixed global multiplier CFG["ewc_lambda"] and accumulates
    the raw (un-normalised) Fisher across tasks — exactly the formulation
    whose scale/order fragility CADRE's M1+M2 are designed to remove.

    Used in Study A (mechanism attribution) as the comparison baseline
    for the EWC consolidation redesign ablation row.

    The penalty is applied in trainer.py as:
        loss += CFG["ewc_lambda"] * ewc.penalty(model)
    """

    def __init__(self):
        self._F:          Dict[str, torch.Tensor] = {}
        self._theta_star: Dict[str, torch.Tensor] = {}

    def consolidate(
        self,
        model:     nn.Module,
        loader:    DataLoader,
        n_batches: int = None,
    ) -> None:
        """
        Estimate raw Fisher and accumulate (no normalisation, no decay).

        Parameters
        ----------
        model     : CADREModel  — model after training on the current modality
        loader    : DataLoader  — training loader
        n_batches : int, optional — overrides CFG["fisher_batches"]
        """
        n_batches = n_batches or CFG["fisher_batches"]
        model.eval()

        F_hat: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in _named_trainable(model)
        }
        count = 0

        for x, y, _ in loader:
            if count >= n_batches:
                break
            x = x.to(CFG["device"])
            y = y.to(CFG["device"])

            model.zero_grad()
            with _autocast():
                loss = F.cross_entropy(model(x), y)
            loss.backward()

            for name, param in _named_trainable(model):
                if param.grad is not None:
                    F_hat[name] += param.grad.detach() ** 2

            count += 1

        for name in F_hat:
            F_hat[name] /= max(count, 1)

        # Accumulate raw Fisher across tasks (no normalisation — vanilla behaviour)
        if not self._F:
            self._F = {name: v.clone() for name, v in F_hat.items()}
        else:
            for name, f_new in F_hat.items():
                if name in self._F:
                    self._F[name] = self._F[name] + f_new
                else:
                    self._F[name] = f_new

        # Reference parameters
        self._theta_star = {
            name: param.detach().clone()
            for name, param in _named_trainable(model)
        }
        model.train()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the raw EWC penalty (λ applied externally by trainer).

        Returns
        -------
        torch.Tensor  scalar on CFG["device"]
        """
        if not self._F:
            return torch.tensor(0.0, device=CFG["device"])

        loss = torch.tensor(0.0, device=CFG["device"])
        for name, param in _named_trainable(model):
            if name in self._F and name in self._theta_star:
                diff = param - self._theta_star[name]
                loss = loss + (self._F[name] * diff ** 2).sum()
        return loss


# =============================================================================
# 4. EMA  (Polyak weight averaging at evaluation time)
# =============================================================================

class EMA:
    """
    Exponential Moving Average of trainable parameters, applied at
    evaluation time only  (SWA-style; Izmailov et al., UAI 2018).

    The shadow weights are maintained throughout training but the live
    model weights are never modified during training — only temporarily
    swapped in for evaluation, then restored for the next training step.

    This avoids the EMA itself reducing measured forgetting by averaging
    away diverged weights; its contribution is attributed separately
    in the ablation study (Study A, "-EMA" row).

    Parameters
    ----------
    decay : float — EMA decay coefficient (default CFG["ema_decay"] = 0.99)
                    shadow ← decay · shadow + (1 − decay) · live_param
    """

    def __init__(self, decay: float = None):
        self.decay   = decay if decay is not None else CFG["ema_decay"]
        self._shadow: Optional[List[torch.Tensor]] = None
        self._backup: Optional[List[torch.Tensor]] = None

    def update(self, params: List[nn.Parameter]) -> None:
        """
        Update the EMA shadow weights with the current live parameters.

        Called once per training step, after the optimiser update.

        Parameters
        ----------
        params : list of nn.Parameter — model.trainable_params()
        """
        if self._shadow is None:
            # Initialise shadow = copy of initial weights
            self._shadow = [p.detach().clone() for p in params]
        else:
            for shadow, param in zip(self._shadow, params):
                # shadow ← decay · shadow + (1 − decay) · param
                shadow.mul_(self.decay).add_(
                    param.detach(), alpha=1.0 - self.decay
                )

    def apply_to(self, params: List[nn.Parameter]) -> bool:
        """
        Temporarily replace live parameters with EMA shadow weights.

        Call before running evaluate(); call restore() immediately after
        to put the live weights back before the next training step.

        Parameters
        ----------
        params : list of nn.Parameter — model.trainable_params()

        Returns
        -------
        bool — True if shadow weights were applied; False if not yet available
               (first modality, before any EMA.update() calls)
        """
        if self._shadow is None:
            return False

        # Back up live weights
        self._backup = [p.detach().clone() for p in params]

        # Overwrite live weights with shadow
        for param, shadow in zip(params, self._shadow):
            param.data.copy_(shadow)

        return True

    def restore(self, params: List[nn.Parameter]) -> None:
        """
        Restore live parameters from the backup saved by apply_to().

        Must be called after every apply_to() that returned True.

        Parameters
        ----------
        params : list of nn.Parameter — model.trainable_params()
        """
        if self._backup is None:
            return
        for param, backup in zip(params, self._backup):
            param.data.copy_(backup)
        self._backup = None

    @property
    def has_shadow(self) -> bool:
        """True after the first EMA.update() call."""
        return self._shadow is not None

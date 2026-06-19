"""
model.py
========
BiomedCLIP backbone loader, LoRA injection, and CADREModel definition.

Architecture summary
--------------------
  Backbone : BiomedCLIP ViT-B/16 visual encoder  — fully FROZEN
  Adapter  : LoRA injected into all attention projection layers (~0.23 % params)
  Head     : Single nn.Linear(embed_dim, 2)       — trainable

Only the LoRA factors (A, B matrices) and the linear head are updated
during continual adaptation. The frozen backbone provides stable
pretrained representations anchored by the prior penalty (see trainer.py).

Reference
---------
  BiomedCLIP: Zhang et al., arXiv:2303.00915 (2023)
  LoRA:       Hu et al., arXiv:2106.09685 (ICLR 2022)
"""

import copy
import math
from typing import List, Optional

import torch
import torch.nn as nn
from open_clip import create_model_from_pretrained, get_tokenizer

from config import CFG


# =============================================================================
# 1. LoRA LINEAR LAYER
# =============================================================================

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with a low-rank additive update.

    Forward pass:
        y = W x  +  (α / r) · B A x

    where W is the original frozen weight matrix,
    A ∈ R^{r × d_in}  is initialised with Kaiming-normal noise,
    B ∈ R^{d_out × r} is initialised to zero (so the adapter starts
    as an identity perturbation and training begins from the pretrained
    representation).

    Parameters
    ----------
    linear : nn.Linear  — the original projection layer to wrap
    rank   : int        — intrinsic rank r of the update (CFG["lora_rank"])
    alpha  : int        — scaling factor α (CFG["lora_alpha"]); output scaled by α/r
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: int):
        super().__init__()

        self.linear  = linear           # frozen original weight; kept for frozen path
        self.rank    = rank
        self.scale   = alpha / rank     # constant multiplier applied to BA

        d_in  = linear.in_features
        d_out = linear.out_features

        # A: random init  (Kaiming-style 1/√r scaling)
        self.A = nn.Parameter(
            torch.randn(rank, d_in) * (1.0 / math.sqrt(rank))
        )
        # B: zero init  → adapter output starts at zero, so W is unchanged
        self.B = nn.Parameter(torch.zeros(d_out, rank))

        # Toggle used by set_lora_enabled() and precompute_frozen_anchor()
        self.enabled: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)                              # base projection
        if self.enabled:
            out = out + self.scale * (x @ self.A.T @ self.B.T)  # LoRA delta
        return out

    def extra_repr(self) -> str:
        d_in  = self.linear.in_features
        d_out = self.linear.out_features
        return (
            f"in={d_in}, out={d_out}, rank={self.rank}, "
            f"scale={self.scale:.3f}, enabled={self.enabled}"
        )


# =============================================================================
# 2. LoRA SET  (container for all injected adapters)
# =============================================================================

class LoRASet(nn.ModuleList):
    """
    Thin nn.ModuleList subclass that holds every LoRALinear in the model.
    Stored as a separate attribute on CADREModel so the trainer can iterate
    over adapters without walking the full parameter tree.
    """
    pass


def set_lora_enabled(lora: LoRASet, flag: bool) -> None:
    """
    Enable or disable all LoRA delta computations in a single call.

    Used by:
      - precompute_frozen_anchor()  → disables LoRA to get g_frozen(x)
      - _train_one_modality()       → re-enables after GradScaler.step()
        (GradScaler can temporarily zero adapters; this restores them)

    Parameters
    ----------
    lora : LoRASet
    flag : bool — True to activate LoRA deltas; False to bypass them
    """
    for adapter in lora:
        adapter.enabled = flag


# =============================================================================
# 3. LoRA INJECTION
# =============================================================================

_ATTN_PROJ_NAMES = ("in_proj", "out_proj", "q_proj", "k_proj", "v_proj")
# open_clip ViT-B/16 attention blocks expose either a fused `in_proj`
# (QKV combined) or separate q/k/v projections depending on the version.
# We attempt all known names and skip those that are absent or not nn.Linear.


def _inject_lora(
    vision_enc: nn.Module,
    rank: int,
    alpha: int,
) -> LoRASet:
    """
    Walk the ViT-B/16 visual encoder and replace every attention projection
    with a LoRALinear wrapper.

    Targets: all transformer residual blocks → attention module →
             {in_proj, out_proj, q_proj, k_proj, v_proj}

    Parameters
    ----------
    vision_enc : nn.Module — the frozen BiomedCLIP visual encoder
    rank       : int       — LoRA rank r
    alpha      : int       — LoRA scale α

    Returns
    -------
    LoRASet
        Flat list of all injected LoRALinear adapters (registered as a
        sub-module so their parameters appear in model.parameters()).
    """
    adapters = LoRASet()

    # open_clip stores transformer blocks under .transformer.resblocks
    try:
        blocks = vision_enc.transformer.resblocks
    except AttributeError:
        raise RuntimeError(
            "Could not locate vision_enc.transformer.resblocks. "
            "Check open_clip version or backbone architecture."
        )

    n_injected = 0
    for block in blocks:
        attn = block.attn
        for proj_name in _ATTN_PROJ_NAMES:
            orig = getattr(attn, proj_name, None)
            if orig is None or not isinstance(orig, nn.Linear):
                continue

            lora = LoRALinear(orig, rank=rank, alpha=alpha)

            # Freeze the original weight (it was already frozen at backbone
            # level, but being explicit avoids subtle grad-graph surprises)
            orig.weight.requires_grad_(False)
            if orig.bias is not None:
                orig.bias.requires_grad_(False)

            setattr(attn, proj_name, lora)
            adapters.append(lora)
            n_injected += 1

    if n_injected == 0:
        raise RuntimeError(
            "LoRA injection found no attention projection layers. "
            "Verify the backbone architecture."
        )

    print(
        f"[_inject_lora] Injected LoRA into {n_injected} projection layers "
        f"(rank={rank}, α={alpha})"
    )
    return adapters


# =============================================================================
# 4. CADRE MODEL
# =============================================================================

class CADREModel(nn.Module):
    """
    BiomedCLIP visual encoder + LoRA adapters + linear classification head.

    Only self.lora parameters and self.head parameters are trainable.
    The vision encoder backbone (self.vision_enc) is permanently frozen.

    Parameters
    ----------
    vision_enc : nn.Module  — frozen BiomedCLIP ViT-B/16 encoder
    lora       : LoRASet    — all injected LoRALinear adapters
    n_classes  : int        — output classes (default 2: binary)
    embed_dim  : int, optional
                            — embedding dimension; auto-detected if None
    """

    def __init__(
        self,
        vision_enc: nn.Module,
        lora: LoRASet,
        n_classes: int = 2,
        embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.vision_enc = vision_enc
        self.lora       = lora

        # Detect embedding dimension from the backbone
        if embed_dim is None:
            embed_dim = self._detect_embed_dim(vision_enc)

        self.head = nn.Linear(embed_dim, n_classes)

        # Initialise head with small weights to avoid saturated softmax
        # at the start of training (especially important for calibration)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _detect_embed_dim(vision_enc: nn.Module) -> int:
        """
        Infer the visual embedding dimension from the encoder.
        Tries common attribute names across open_clip versions.
        """
        for attr in ("output_dim", "embed_dim", "width"):
            val = getattr(vision_enc, attr, None)
            if isinstance(val, int):
                return val

        # Fallback: run a dummy forward pass on CPU
        dummy = torch.zeros(1, 3, CFG["img_size"], CFG["img_size"])
        with torch.no_grad():
            out = vision_enc(dummy.cpu())
        if isinstance(out, (list, tuple)):
            out = out[0]
        dim = out.shape[-1]
        print(f"[CADREModel] embed_dim auto-detected via dummy forward: {dim}")
        return dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, 3, H, W)

        Returns
        -------
        torch.Tensor  shape (B, n_classes) — raw logits (pre-softmax)
        """
        feats = self.vision_enc(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]          # take image embedding (not text path)
        return self.head(feats)

    # ── Parameter utilities ───────────────────────────────────────────────────

    def trainable_params(self) -> List[nn.Parameter]:
        """Return only the parameters that require gradients (LoRA + head)."""
        return [p for p in self.parameters() if p.requires_grad]

    def param_efficiency(self) -> float:
        """
        Percentage of total parameters that are trainable.

        Expected: ~0.23 % for LoRA rank-8 on ViT-B/16 (25 attention layers,
        in_proj + out_proj) plus the linear head.
        """
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_params())
        pct = 100.0 * trainable / max(total, 1)
        return pct

    def param_counts(self) -> dict:
        """Return a dict with total, trainable, and frozen parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_params())
        return {
            "total":     total,
            "trainable": trainable,
            "frozen":    total - trainable,
            "pct":       100.0 * trainable / max(total, 1),
        }


# =============================================================================
# 5. BACKBONE LOADER
# =============================================================================

def load_backbone():
    """
    Download (or load from cache) the BiomedCLIP backbone and return the
    frozen visual encoder, tokenizer, and image preprocessor.

    Returns
    -------
    vision_enc  : nn.Module   — ViT-B/16 visual encoder; all params frozen
    tokenizer   : callable    — BiomedCLIP text tokenizer (API parity only;
                                visual-only path does not use text encoding)
    preprocess  : callable    — torchvision image transform expected by ViT-B/16
                                (resize → centre-crop → normalise)

    Notes
    -----
    The backbone is loaded once in run_experiments.py and passed by reference
    to build_model(); it is deepcopied inside build_model() so every
    experimental run gets an independent copy without re-downloading.
    """
    print(f"[load_backbone] Loading {CFG['backbone_name']} ...")
    model, preprocess = create_model_from_pretrained(CFG["backbone_name"])
    tokenizer         = get_tokenizer(CFG["backbone_name"])

    vision_enc = model.visual.eval()

    # Freeze every parameter in the backbone
    for param in vision_enc.parameters():
        param.requires_grad_(False)

    param_count = sum(p.numel() for p in vision_enc.parameters())
    print(
        f"[load_backbone] Backbone loaded. "
        f"Visual encoder params: {param_count:,} (all frozen)."
    )
    return vision_enc, tokenizer, preprocess


# =============================================================================
# 6. MODEL FACTORY
# =============================================================================

def build_model(
    backbone: nn.Module,
    tok,                        # tokenizer — unused in visual path; kept for API
    use_lora: bool = True,
) -> CADREModel:
    """
    Construct a fresh CADREModel for one experimental run.

    Deep-copies the backbone so each run is fully independent.
    Injects LoRA (if use_lora=True) using CFG["lora_rank"] and
    CFG["lora_alpha"], then freezes the backbone and activates
    gradients only on LoRA factors and the head.

    Parameters
    ----------
    backbone : nn.Module — frozen visual encoder from load_backbone()
    tok      : any       — tokenizer (passed through for API parity)
    use_lora : bool      — if False, returns a linear-probe model (no LoRA)

    Returns
    -------
    CADREModel
        Ready-to-train model on CFG["device"]. Trainable parameters:
          - LoRA A and B matrices (all attention projections)
          - Linear head weights and bias
        Everything else: frozen.
    """
    # Deep copy so runs do not share parameters
    enc = copy.deepcopy(backbone).to(CFG["device"])

    # Ensure backbone is fully frozen after copy
    for param in enc.parameters():
        param.requires_grad_(False)

    # Inject LoRA adapters
    if use_lora:
        lora = _inject_lora(enc, rank=CFG["lora_rank"], alpha=CFG["lora_alpha"])
    else:
        lora = LoRASet()   # empty — linear probe baseline

    # Build full model
    model = CADREModel(enc, lora).to(CFG["device"])

    # Activate gradients: LoRA factors + head only
    for adapter in lora:
        for param in adapter.parameters():
            # Only A and B should be trainable; the wrapped linear stays frozen
            if param is adapter.A or param is adapter.B:
                param.requires_grad_(True)

    for param in model.head.parameters():
        param.requires_grad_(True)

    # Print parameter efficiency
    counts = model.param_counts()
    print(
        f"[build_model] CADREModel ready | "
        f"trainable: {counts['trainable']:,} / {counts['total']:,} "
        f"({counts['pct']:.3f} %)"
    )

    return model

"""
================================================================================
 src/model.py: YOUR IMPLEMENTATION GOES HERE (learning exercise)
================================================================================

This file is a SKELETON. Every function/class below has its final signature so that
`src/train.py` and `tests/test_pipeline.py` can import it, but each body raises
NotImplementedError. Work through them in order:

    1. CalibratedMTGPredictor   the network                     (≈ 15 lines)
    2. brier_score_loss         the training loss                (≈ 4 lines)
    3. expected_calibration_error   the calibration metric         (≈ 15 lines)
    4. compute_flip_ev          turning probabilities into a decision (≈ 8 lines)

When you finish one, run:
    pytest tests/test_pipeline.py -k model -v
The model tests skip themselves while a function still raises NotImplementedError and
start running (and must pass) as soon as you implement it.

Then train for real:
    python -m src.train train
    python -m src.train predict --top 20

Class convention used everywhere in the project (see src/features.py):
    index 0 = Drop  (forward return <= -20%)
    index 1 = Stable
    index 2 = Spike (forward return >= +20%)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401  (you'll want F.softmax / F.one_hot / F.gelu)

N_CLASSES = 3
CLASS_NAMES = ["Drop >20%", "Stable", "Spike >20%"]


# =============================================================================
# 1. The network
# =============================================================================
class CalibratedMTGPredictor(nn.Module):
    """Small MLP mapping a card's feature vector to 3-class logits.

    Architecture (what you are building):

        x  (batch, in_features)
        │
        ├─ Linear(in_features → 64)
        ├─ GELU
        ├─ Dropout(p = 0.15)
        ├─ Linear(64 → 32)
        ├─ GELU
        └─ Linear(32 → 3)        → logits (batch, 3)   [Drop, Stable, Spike]

    Step-by-step:

      __init__
        a) Call `super().__init__()` FIRST. nn.Module needs it to register parameters.
        b) Store `in_features` and `n_classes` on self (handy for checkpoints/debugging).
        c) Build the layers. Two common styles, pick one:
             - an `nn.Sequential(...)` holding all six layers in order, or
             - named attributes (self.fc1, self.act1, self.drop, self.fc2, ...).
           Any nn.Module assigned as an attribute (or inside Sequential) is
           automatically registered, so `model.parameters()` will find it.
        d) Use nn.GELU() (smooth ReLU: GELU(x) = x · Φ(x), where Φ is the standard
           normal CDF) and nn.Dropout(dropout).
        e) The final Linear has NO activation. Return raw logits z ∈ ℝ³.
           Softmax is applied by the loss / at inference time. Putting softmax inside
           the network and then again in the loss is a classic bug.

      forward
        f) Pass x through the layers and return the logits tensor of shape
           (batch, n_classes). dtype should stay float32.

    Why dropout only after the first hidden layer? It's the widest layer (most
    capacity to overfit noise). Remember: dropout is active in `model.train()` and
    disabled in `model.eval()`. train.py handles switching for you.

    Optional extra once it works: initialise the last layer's bias to
    log(class_priors) so the untrained model already predicts the base rates
    (a well-known trick that speeds up training on imbalanced classes).
    """

    def __init__(
        self,
        in_features: int,
        hidden1: int = 64,
        hidden2: int = 32,
        dropout: float = 0.15,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__()
        # TODO (steps a–e): store sizes and define your layers here.
        raise NotImplementedError("TODO: implement me")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_features) float32  →  logits: (batch, n_classes)."""
        # TODO (step f)
        raise NotImplementedError("TODO: implement me")


# =============================================================================
# 2. The loss: multi-class Brier score
# =============================================================================
def brier_score_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean multi-class Brier score, differentiable w.r.t. `logits`.

    Args:
        logits:  (N, K) raw network outputs (float)
        targets: (N,)   integer class labels in {0, …, K-1} (torch.long)
    Returns:
        scalar tensor. 0 = perfect, and it can't exceed 2 (confidently wrong).

    Formula:
                    1   N   K
        Brier  =  ───  Σ   Σ  ( p_ik − y_ik )²
                    N  i=1 k=1

        where p_ik = softmax(z_i)_k = exp(z_ik) / Σ_j exp(z_ij)
              y_ik = 1 if targets[i] == k else 0      (one-hot)

    Steps:
      1. probs = softmax over the CLASS dimension (dim=1, not dim=0!)  → (N, K)
      2. one_hot = one-hot encode targets with K = logits.shape[1] classes, then
         cast to probs.dtype (F.one_hot returns int64, and float - int fails/upcasts).
      3. Squared error per element → sum over classes (dim=1) → mean over the batch.

    Sanity checks (the tests use these):
      - Perfect, confident predictions        → ≈ 0
      - Uniform probs (1/3 each), any target  → (2/3)² + 2·(1/3)² = 2/3
      - Must have a gradient: loss.backward() should populate logits.grad.

    Why Brier instead of cross-entropy? Both are *strictly proper scoring rules*:
    if the true probability of an outcome is q, the expected score is uniquely
    minimised by predicting exactly p = q. Proof sketch for one class:
        E[(p − Y)²] = q(p − 1)² + (1 − q)p²,   d/dp = 2(p − q) = 0  ⇒  p = q.
    So the model is never rewarded for being over- or under-confident. Its best
    strategy is to report its honest belief, i.e. be *calibrated*. Brier is also
    bounded (unlike log-loss, which explodes to ∞ on a single confident miss), which
    makes it robust to the noisy, fat-tailed labels you get from card prices.
    Trade-off: its gradients are weaker for very wrong predictions, so it can train
    slower than cross-entropy. Watch the learning curves.
    """
    # TODO: steps 1–3
    raise NotImplementedError("TODO: implement me")


# =============================================================================
# 3. The calibration metric: Expected Calibration Error (ECE)
# =============================================================================
def expected_calibration_error(
    probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10
) -> float:
    """Top-label Expected Calibration Error.

    Args:
        probs:   (N, K) predicted probabilities (rows sum to 1, i.e. ALREADY softmaxed)
        targets: (N,)   integer labels
        n_bins:  number of equal-width confidence bins over [0, 1]
    Returns:
        python float in [0, 1]. 0 = perfectly calibrated.

    Intuition: among all predictions made with ~70% confidence, about 70% should be
    correct. ECE measures how far that's violated, on average.

    Steps:
      1. confidence_i = max_k p_ik          (the model's top probability)
         prediction_i = argmax_k p_ik
         correct_i    = 1.0 if prediction_i == targets_i else 0.0
         (torch.max(probs, dim=1) gives you both values and indices at once.)
      2. Bin edges: torch.linspace(0, 1, n_bins + 1) → bins (0, 0.1], (0.1, 0.2], …
         Use  lo < confidence <= hi  so a confidence of exactly 1.0 lands in the
         last bin. (Top-label confidence is always ≥ 1/K, so 0 can't occur.)
      3. For each bin B_b that is NON-EMPTY (skip empty bins: dividing by 0 → NaN):
             acc(B_b)  = mean of correct_i    for i in B_b
             conf(B_b) = mean of confidence_i for i in B_b
             weight_b  = |B_b| / N
      4. Combine:
                    n_bins
             ECE  =   Σ    ( |B_b| / N ) · | acc(B_b) − conf(B_b) |
                     b=1
      5. Return `float(ece)`. This is a metric, not a loss: wrap the body in
         `with torch.no_grad():` and don't try to backprop through it (the binning
         is non-differentiable anyway).

    Sanity checks:
      - Every prediction confidence 1.0 and correct → ECE = 0
      - Every prediction confidence 0.9 but only 50% correct → ECE = 0.4

    Caveat worth knowing: on a heavily imbalanced problem (most cards are "Stable"),
    a model that always says "Stable, 90%" can score a low ECE while being useless
    for finding spikes. That's why train.py logs Brier, ECE, AND per-class stats.
    """
    # TODO: steps 1–5
    raise NotImplementedError("TODO: implement me")


# =============================================================================
# 4. The decision: post-fee expected value of a flip
# =============================================================================
def compute_flip_ev(
    probs: list[float],
    current_price: float,
    expected_gain: float = 0.25,
    expected_loss: float = 0.20,
    fee_rate: float = 0.14,
) -> float:
    """Expected dollar profit of buying a copy at `current_price` today and selling it
    at the end of the prediction horizon on a marketplace that keeps `fee_rate`.

    Args:
        probs:          [p_drop, p_stable, p_spike]  (sum to 1; order = class indices)
        current_price:  P, what you pay now (USD)
        expected_gain:  g, average % rise *given* a spike   (default +25%)
        expected_loss:  l, average % fall *given* a drop    (default −20%)
        fee_rate:       f, marketplace + payment fees as a fraction of the sale
                        (TCGplayer ≈ 10.25% commission + ≈ 3.5% processing ≈ 14%)
    Returns:
        EV in dollars per copy (float). Positive = profitable on average.

    The three scenarios. You buy at P; you receive (1 − f) × sale price:

        Spike:   sale = P(1 + g)  →  profit_spike  = P(1 + g)(1 − f) − P
        Stable:  sale = P         →  profit_stable = P(1 − f)        − P   = −fP
        Drop:    sale = P(1 − l)  →  profit_drop   = P(1 − l)(1 − f) − P

    Expected value (law of total expectation):

        EV = p_spike · profit_spike + p_stable · profit_stable + p_drop · profit_drop

    which simplifies (since p_drop + p_stable + p_spike = 1) to

        EV = P · [ (1 − f) · (1 + p_spike·g − p_drop·l) − 1 ]

    Implement it either way. Writing out the three scenarios first is clearer;
    then check your code against the simplified form.

    Steps:
      1. Unpack: p_drop, p_stable, p_spike = probs   (validate len == 3; optionally
         assert abs(sum(probs) − 1) < 1e-4 to catch un-softmaxed logits)
      2. Compute the three per-scenario profits above.
      3. Return the probability-weighted sum as a float.

    Sanity checks:
      - probs = [0, 1, 0], P = $10, f = 0.14  →  EV = −$1.40 (stable = you just pay fees)
      - Break-even spike probability (p_drop = 0) solves (1 − f)(1 + p·g) = 1:
            p* = (1/(1 − f) − 1) / g = (1/0.86 − 1)/0.25 ≈ 0.651
        i.e. with 14% fees you need ~65% confidence in a +25% spike just to break
        even. That's why *calibration* matters more than accuracy here: an
        overconfident model would turn this inequality into real losses.

    Extensions to try later: subtract shipping (a flat $ amount, which kills EV on
    cheap cards and is another reason the pipeline ignores sub-$3 cards), or use the
    model's own conditional mean return per class instead of fixed g and l.
    """
    # TODO: steps 1–3
    raise NotImplementedError("TODO: implement me")

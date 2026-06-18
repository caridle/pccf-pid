"""
Event-triggered dynamic PCCF controller with entropy spike detection.

Key insight: precision modulation should be event-driven, not continuous.
The controller maintains normal operation (pi=1.0) and only intervenes
when prediction entropy spikes relative to its recent baseline,
indicating a genuine regime change / prior conflict.

Algorithm:
  1. Maintain exponential moving average of token-by-token entropy (baseline)
  2. On each token, compare current entropy to baseline
  3. If entropy > threshold * baseline AND sustained for N tokens:
     → trigger: pi drops to pi_min, stay for hold_duration tokens
  4. After hold_duration, gradually recover to pi_max
  5. Hysteresis prevents re-triggering during recovery
"""

import math
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SpikeDetector:
    """Detect entropy spikes relative to a moving baseline.

    Uses z-score-like detection: triggers when current entropy
    exceeds baseline by k standard deviations.
    """
    alpha: float = 0.02          # baseline EMA smoothing (smaller = slower)
    threshold_sigma: float = 3.0 # sigma multiplier for trigger
    min_entropy: float = 0.1     # ignore very low entropy (deterministic tokens)
    warmup_steps: int = 8        # steps to establish baseline before detecting

    # State
    baseline_mean: float = field(default=0.0, init=False)
    baseline_var: float = field(default=0.0, init=False)
    step_count: int = field(default=0, init=False)
    recent_history: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self.baseline_mean = 0.0
        self.baseline_var = 0.01
        self.step_count = 0
        self.recent_history = []

    def update(self, entropy: float) -> dict:
        """Process an entropy observation. Returns detection status."""
        self.step_count += 1
        e = float(entropy)

        # Warmup: collect samples before computing baseline
        if self.step_count <= self.warmup_steps:
            self.recent_history.append(e)
            if self.step_count == self.warmup_steps:
                self.baseline_mean = np.mean(self.recent_history)
                self.baseline_var = max(np.var(self.recent_history), 0.001)
            return {"spike": False, "sigma": 0.0, "baseline": self.baseline_mean}

        # Update baseline with EMA
        self.baseline_mean = (1 - self.alpha) * self.baseline_mean + self.alpha * e
        dev = (e - self.baseline_mean) ** 2
        self.baseline_var = (1 - self.alpha) * self.baseline_var + self.alpha * dev

        sigma = math.sqrt(max(self.baseline_var, 1e-6))
        z_score = (e - self.baseline_mean) / sigma if sigma > 0 else 0.0

        spike = z_score > self.threshold_sigma and e > self.min_entropy

        return {"spike": spike, "sigma": z_score, "baseline": self.baseline_mean}


@dataclass
class EventTriggeredPCCF:
    """Event-triggered PCCF controller.

    Normally operates at pi=1.0 (standard attention).
    When the spike detector fires, precision drops to pi_min
    for hold_steps tokens, then recovers.
    """
    pi_min: float = 0.1            # precision floor when triggered
    pi_max: float = 1.0            # normal precision
    hold_steps: int = 20           # tokens to hold low precision after trigger
    recovery_steps: int = 20       # tokens to recover back to pi_max
    cooldown_steps: int = 40       # minimum steps between triggers
    spike_detector: SpikeDetector = None

    # State
    pi: float = field(default=1.0, init=False)
    state: str = field(default="normal", init=False)  # normal | triggered | holding | recovering | cooldown
    state_counter: int = field(default=0, init=False)
    trigger_count: int = field(default=0, init=False)
    trigger_log: list = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.spike_detector is None:
            self.spike_detector = SpikeDetector()
        self.pi = self.pi_max
        self.state = "normal"
        self.state_counter = 0
        self.trigger_count = 0
        self.trigger_log = []

    def step(self, entropy: float) -> dict:
        """
        Process one token's entropy. Returns {pi, state, triggered}.

        State machine:
          normal  --[spike detected]--> triggered
          triggered --[set pi=pi_min]--> holding
          holding --[hold_steps elapsed]--> recovering
          recovering --[pi reaches pi_max]--> cooldown
          cooldown --[cooldown_steps elapsed]--> normal
        """
        detection = self.spike_detector.update(entropy)

        if self.state == "normal":
            if detection["spike"]:
                self.state = "triggered"
                self.state_counter = 0
                self.trigger_count += 1
                self.trigger_log.append({
                    "step": self.spike_detector.step_count,
                    "entropy": round(entropy, 4),
                    "sigma": round(detection["sigma"], 1),
                    "baseline": round(detection["baseline"], 4),
                })

        if self.state == "triggered":
            self.pi = self.pi_min
            self.state = "holding"
            self.state_counter = 0

        elif self.state == "holding":
            self.pi = self.pi_min
            self.state_counter += 1
            if self.state_counter >= self.hold_steps:
                self.state = "recovering"
                self.state_counter = 0

        elif self.state == "recovering":
            # Linear ramp back to pi_max
            progress = min(self.state_counter / max(self.recovery_steps, 1), 1.0)
            self.pi = self.pi_min + (self.pi_max - self.pi_min) * progress
            self.state_counter += 1
            if progress >= 1.0:
                self.state = "cooldown"
                self.state_counter = 0

        elif self.state == "cooldown":
            self.pi = self.pi_max
            self.state_counter += 1
            if self.state_counter >= self.cooldown_steps:
                self.state = "normal"
                self.state_counter = 0

        else:
            self.pi = self.pi_max
            self.state_counter += 1

        return {
            "pi": self.pi,
            "state": self.state,
            "triggered": self.state in ("triggered", "holding"),
            "detection": detection,
        }

    def reset(self):
        self.spike_detector.__post_init__()
        self.pi = self.pi_max
        self.state = "normal"
        self.state_counter = 0
        self.trigger_count = 0
        self.trigger_log.clear()


def compute_entropy_from_logits(logits):
    """Compute entropy from logits tensor (batch, vocab) or (batch, seq, vocab)."""
    import torch
    import torch.nn.functional as F
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return float(entropy.item())

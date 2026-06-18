"""
PCCF precision-modulated attention for HuggingFace Transformer models.

Strategy: Modify attention scaling factor (temperature control).

The attention computation in Qwen2 is:
  Attention = softmax(QK^T * scaling)

Where scaling = 1/sqrt(head_dim).

PCCF modifies this to:
  Attention = softmax(QK^T * scaling * pi)

When pi < 1, attention logits are compressed -> softer attention distribution.
When pi = 1, standard behavior.
When pi > 1, attention is sharpened.

This implements precision as an effective temperature of attention:
- Low pi (e.g., 0.1): temperature = 10, flat attention -> less reliance on strong priors
- High pi (e.g., 1.0): temperature = 1, sharp attention -> standard behavior

Works with SDPA, eager, and flash_attn implementations.
Tested with Qwen2.5-1.5B-Instruct (transformers 5.9.0).
"""


# ── Model patching / restoring utilities ───────────────────────────────

_ORIGINAL_ATTR = "_pccf_original_scaling"


def _parse_layer_index(name: str):
    parts = name.split(".")
    for marker in ("layers", "h", "block"):
        if marker not in parts:
            continue
        idx = parts.index(marker) + 1
        if idx < len(parts) and parts[idx].isdigit():
            return int(parts[idx])
    return None


def _is_target_attention_module(module) -> bool:
    cls_name = module.__class__.__name__
    return "Attention" in cls_name and hasattr(module, "scaling") and hasattr(module, "q_proj")


def _selected_layer_indices(layer_indices: list[int], layer_mode: str):
    if layer_mode == "all":
        return set(layer_indices)
    n_layers = len(layer_indices)
    if n_layers == 0:
        return set()
    sorted_layers = sorted(layer_indices)
    if layer_mode == "early":
        return set(sorted_layers[: max(1, n_layers // 3)])
    if layer_mode == "middle":
        start = n_layers // 3
        end = max(start + 1, 2 * n_layers // 3)
        return set(sorted_layers[start:end])
    if layer_mode == "late":
        return set(sorted_layers[2 * n_layers // 3 :])
    if layer_mode.startswith("last"):
        count_text = layer_mode.removeprefix("last")
        count = int(count_text) if count_text else 1
        return set(sorted_layers[-max(1, count):])
    if layer_mode.startswith("layers:"):
        values = layer_mode.removeprefix("layers:")
        return {int(x.strip()) for x in values.split(",") if x.strip()}
    raise ValueError(
        f"Unsupported layer_mode={layer_mode!r}. "
        "Use all, early, middle, late, lastN, or layers:i,j,k."
    )


def patch_model_attention(model, precision_pi: float = 1.0, layer_mode: str = "all"):
    """
    Modify all attention layers to apply PCCF precision weighting
    by scaling the attention temperature.

    This modifies self.scaling on each attention module:
      new_scaling = original_scaling * precision_pi

    Args:
        model: HuggingFace causal LM (e.g., Qwen2ForCausalLM).
        precision_pi: Precision weight. <1 = flatter attention, >1 = sharper.

    Returns:
        Number of modules modified.
    """
    targets = []
    for name, module in model.named_modules():
        if not _is_target_attention_module(module):
            continue
        layer_idx = _parse_layer_index(name)
        targets.append((name, module, layer_idx))

    indexed_layers = sorted({idx for _, _, idx in targets if idx is not None})
    selected_layers = _selected_layer_indices(indexed_layers, layer_mode)

    count = 0
    for name, module, layer_idx in targets:
        if layer_idx is not None and layer_idx not in selected_layers:
            continue
        if layer_idx is None and layer_mode != "all":
            continue
        if not hasattr(module, _ORIGINAL_ATTR):
            setattr(module, _ORIGINAL_ATTR, module.scaling)
        module.scaling = getattr(module, _ORIGINAL_ATTR) * precision_pi
        module._pccf_pi = precision_pi
        module._pccf_layer_mode = layer_mode
        count += 1

    print(f"[PCCF] Modified scaling on {count} attention modules (pi={precision_pi}, layer_mode={layer_mode})")
    return count


def update_pi_all(model, new_pi: float):
    """
    Update precision weight on all modified attention modules at runtime.
    Restores original scaling first, then applies new pi.
    """
    count = 0
    for _, module in model.named_modules():
        if not _is_target_attention_module(module):
            continue
        if hasattr(module, _ORIGINAL_ATTR):
            module.scaling = getattr(module, _ORIGINAL_ATTR) * new_pi
            module._pccf_pi = new_pi
            count += 1
    return count


def restore_model_attention(model):
    """Restore original scaling values on all attention modules."""
    for _, module in model.named_modules():
        if not _is_target_attention_module(module):
            continue
        if hasattr(module, _ORIGINAL_ATTR):
            module.scaling = getattr(module, _ORIGINAL_ATTR)
            delattr(module, _ORIGINAL_ATTR)
        if hasattr(module, "_pccf_pi"):
            delattr(module, "_pccf_pi")
    print("[PCCF] Restored original attention scaling")

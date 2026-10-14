"""Independent NumPy RoPE reference preserved from the validated implementation."""
import numpy as np


def _llama3_scale_freqs(freqs, factor=32.0, low=1.0, high=4.0, old_ctx=8192):
    """Llama-3 RoPE frequency scaling (smooth low/high-frequency interpolation)."""
    out = np.empty_like(freqs)
    low_wl = old_ctx / low                                     # wavelength thresholds
    high_wl = old_ctx / high
    for i, f in enumerate(freqs):
        wl = 2.0 * np.pi / f
        if wl > low_wl:
            out[i] = f / factor                               # low freq: scale down
        elif wl < high_wl:
            out[i] = f                                        # high freq: unchanged
        else:
            s = (old_ctx / wl - low) / (high - low)           # smooth interpolation
            out[i] = (1 - s) * f / factor + s * f
    return out


def rope_freqs_row(hd, base=10000.0, llama3_scaling=False):
    """Half-duplicated frequency row [1,hd] for on-device RoPE: angle = pos * freqs.
    Static constant (only `pos` varies at runtime), so decode can feed just the
    position and the NPU builds cos/sin itself."""
    half = hd // 2
    freqs = base ** (-2.0 * np.arange(half) / hd)
    if llama3_scaling:
        freqs = _llama3_scale_freqs(freqs)
    return np.concatenate([freqs, freqs])[None, :]

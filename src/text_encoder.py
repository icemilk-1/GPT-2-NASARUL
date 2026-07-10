"""Text encoder for C-MAPSS sensor data → natural language prompts.

Converts a sliding window of sensor readings into a compact text description
suitable for GPT-2 tokenization (target: 300-500 tokens per window).
"""

from __future__ import annotations

import numpy as np

# =============================================================================
#  Sensor name mapping (C-MAPSS turbofan engine sensors)
# =============================================================================

SENSOR_NAMES: dict[str, str] = {
    "s1":  "fan inlet temp",
    "s2":  "LPC outlet temp",
    "s3":  "HPC outlet temp",
    "s4":  "LPT outlet temp",
    "s5":  "fan inlet pressure",
    "s6":  "bypass duct pressure",
    "s7":  "HPC outlet pressure",
    "s8":  "fan speed",
    "s9":  "core speed",
    "s10": "engine pressure ratio",
    "s11": "HPC static pressure",
    "s12": "fuel-air ratio",
    "s13": "corrected fan speed",
    "s14": "corrected core speed",
    "s15": "bypass ratio",
    "s16": "burner fuel-air ratio",
    "s17": "bleed enthalpy",
    "s18": "demanded fan speed",
    "s19": "demanded corrected fan speed",
    "s20": "HPT coolant bleed",
    "s21": "LPT coolant bleed",
    "cond1": "flight altitude",
    "cond2": "Mach number",
    "cond3": "throttle",
}

SETTING_NAMES: dict[str, str] = {
    "cond1": "altitude",
    "cond2": "Mach",
    "cond3": "throttle",
}

# =============================================================================
#  Text prompt builder
# =============================================================================

def _trend_desc(slope: float) -> str:
    """Describe trend direction and strength."""
    if abs(slope) < 0.001:
        return "stable"
    direction = "up" if slope > 0 else "down"
    if abs(slope) > 0.05:
        return f"{direction} fast"
    elif abs(slope) > 0.01:
        return f"{direction}"
    else:
        return f"{direction} slow"


def window_to_text(
    window: np.ndarray,
    feature_cols: list[str],
) -> str:
    """Convert a sensor window (T, F) to a compact natural-language prompt.

    Target: ~350-450 tokens for 14 sensors over 30 time steps.
    """
    T, _ = window.shape
    sensor_cols = [c for c in feature_cols if c.startswith("s")]
    cond_cols = [c for c in feature_cols if c.startswith("cond")]

    parts: list[str] = []

    # ---- Operating condition (from cond features if available) ----
    if cond_cols:
        vals = [f"{SETTING_NAMES.get(c,c)}={window[-1, feature_cols.index(c)]:.2f}" for c in cond_cols]
        parts.append(f"Engine: {', '.join(vals)}.")

    # ---- Sensor trends (compact, one line per sensor) ----
    parts.append(f"Sensors(t={T}):")
    for col in sensor_cols:
        idx = feature_cols.index(col)
        series = window[:, idx]
        latest = float(series[-1])
        slope = float(np.polyfit(np.arange(T), series, 1)[0])

        # Compact trend token
        if slope > 0.01:
            arrow = "up"
        elif slope < -0.01:
            arrow = "down"
        else:
            arrow = "flat"

        name = SENSOR_NAMES.get(col, col)
        parts.append(f"{col}({name})={latest:.1f} {arrow}")

    # ---- Query ----
    parts.append("Q: remaining useful life in cycles?")
    parts.append("A:")

    return "\n".join(parts)


def windows_to_texts(
    X: np.ndarray,
    feature_cols: list[str],
) -> list[str]:
    """Convert a batch of windows (N, T, F) to text prompts."""
    texts = []
    for i in range(X.shape[0]):
        texts.append(window_to_text(X[i], feature_cols))
    return texts

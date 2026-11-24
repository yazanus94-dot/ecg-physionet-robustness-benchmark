"""Starter solution for the ECG denoising benchmark task.

Replace the body of ``denoise`` with a real implementation. The grader checks
signal quality, edge cases and determinism - returning the input unchanged
will fail.
"""

from __future__ import annotations

import numpy as np


def denoise(signal: np.ndarray, fs: float) -> np.ndarray:
    """Return a denoised copy of the ECG signal.

    Requirements:
      - non-empty 1-D finite float array, same shape as the input;
      - removes baseline wander and out-of-band noise without shifting R peaks;
      - raises ValueError for empty, 2-D, non-finite, too-short input or
        non-positive fs;
      - deterministic and linear in the input amplitude.
    """
    raise NotImplementedError("Implement the denoising reference approach")

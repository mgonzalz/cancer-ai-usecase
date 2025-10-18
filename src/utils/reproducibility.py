# Global determinism utilities for TensorFlow
# Purpose:
# - Ensure fully reproducible training runs by fixing all random seeds.
# - Configure TensorFlow to use deterministic operations whenever possible,
#   avoiding non-deterministic GPU kernels or parallel ops.
# - Useful for experiments, model comparisons, and debugging.
#
# Notes:
# - Setting TF_DETERMINISTIC_OPS="1" may slightly reduce GPU performance,
#   but guarantees bitwise reproducibility across runs.
# - Works best with TensorFlow 2.10.

import os
import random

import numpy as np
import tensorflow as tf


def set_global_determinism(seed: int = 1337) -> None:
    """
    Sets global random seeds and configures TensorFlow for deterministic behavior (when possible).
    This helps ensure reproducibility of experiments across different runs by
    fixing all sources of randomness (Python, NumPy, and TensorFlow).
    Args:
        - seed: Integer value used to initialize all random generators.
    """
    # Set Python's built-in hash seed (affects hashing-based randomization)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Enforce deterministic TensorFlow operations when supported
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[OK] Determinism configured with seed={seed}")

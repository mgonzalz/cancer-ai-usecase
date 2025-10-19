# Quick GPU and configuration smoke test for TensorFlow.
# Purpose:
# - Detects available GPUs and verifies that TensorFlow can access them.
# - Loads the main YAML configuration file (config/base.yaml).
# - Sets global determinism to ensure reproducible training behavior.
# - Prints key configuration values (image size, batch size, Azure container, etc.).
#
# Notes:
# - This script is meant as a lightweight environment sanity check.
# - Useful before running full training to confirm GPU + config setup.

import tensorflow as tf

from src.utils.config import load_config
from src.utils.reproducibility import set_global_determinism

print("\n=== GPU Detection ===")
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for i, g in enumerate(gpus):
        print(f"Available GPU {i}: {g.name}")
    print(f"[OK] {len(gpus)} Physical GPU(s) detected.\n")
else:
    print("[WARN] No GPU detected. Using CPU.\n")

cfg = load_config("config/base.yaml")
set_global_determinism(cfg["seed"])
print(
    f"Config loaded successfully:\n  img_size={cfg['img_size']}  batch={cfg['batch_size']}"
)
print(f"Azure container: {cfg['azure']['container_raw']}")

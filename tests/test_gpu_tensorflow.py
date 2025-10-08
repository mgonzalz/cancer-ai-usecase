import os
import unittest

import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Lower verbosity for tests


# Ensure TensorFlow detects the GPU (if available)
class TestTensorFlowGPU(unittest.TestCase):
    # Check if any GPU is visible to TensorFlow.
    def test_gpu_is_visible(self):
        gpus = tf.config.list_physical_devices("GPU")
        self.assertTrue(len(gpus) > 0, f"No GPUs detected. Devices: {gpus}")

    # Detects if TensorFlow is actually running ops on GPU (not just seeing it).
    def test_tensor_ops_on_gpu(self):
        # Matmul on GPU and device verification
        with tf.device("/GPU:0"):
            a = tf.random.uniform((512, 512))
            b = tf.random.uniform((512, 512))
            c = tf.matmul(a, b)
        self.assertEqual(c.shape, (512, 512))
        # In eager mode, c.device contains the physical device
        self.assertIn("GPU", c.device, f"Tensor not executed on GPU: {c.device}")

    # Small Keras model (2 dense layers) and train for 1 epoch with random data on GPU.
    def test_keras_one_train_step_on_gpu(self):
        import numpy as np

        x = np.random.randn(1024, 32).astype("float32")
        y = np.random.randn(1024, 1).astype("float32")

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu", input_shape=(32,)),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")

        with tf.device("/GPU:0"):
            hist = model.fit(x, y, epochs=1, batch_size=64, verbose=0)
        self.assertIn("loss", hist.history)
        self.assertTrue(hist.history["loss"][0] >= 0.0)

    # Check CUDA/cuDNN integration via build info or memory query.
    def test_cuda_build_info_or_memory(self):
        # Trying to get build/memory info to confirm CUDA/cuDNN integration.
        build = getattr(tf.sysconfig, "get_build_info", lambda: {})()
        # Not all versions expose the same info; accept any of the signals.
        signals = []
        if isinstance(build, dict):
            cuda = build.get("cuda_version") or build.get("cuda_version_number")
            cudnn = build.get("cudnn_version")
            signals.extend([cuda, cudnn])

        ok_signal = any(bool(s) for s in signals)

        # Alternative: query memory (may not exist in older TF versions).
        mem_ok = False
        try:
            info = tf.config.experimental.get_memory_info("GPU:0")
            mem_ok = isinstance(info, dict) and "current" in info
        except Exception:
            mem_ok = False

        self.assertTrue(
            ok_signal or mem_ok,
            f"Could not confirm CUDA/cuDNN or GPU memory. build={build}",
        )


if __name__ == "__main__":
    unittest.main()

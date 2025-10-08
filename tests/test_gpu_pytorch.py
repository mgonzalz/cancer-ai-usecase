import unittest

import torch
import torch.nn as nn


# Ensure PyTorch detects the GPU (if available)
class TestPyTorchGPU(unittest.TestCase):
    # Check if CUDA is available.
    def test_cuda_available(self):
        self.assertTrue(torch.cuda.is_available(), "CUDA not available in PyTorch.")

    # Check if cuDNN is available (for performance/compatibility).
    def test_cudnn_available(self):
        self.assertTrue(
            torch.backends.cudnn.is_available(),
            "cuDNN not available in PyTorch (performance/compatibility).",
        )

    # Try a tensor operation on GPU because just seeing the device is not enough.
    def test_tensor_ops_on_cuda(self):
        device = torch.device("cuda:0")
        a = torch.randn(512, 512, device=device)
        b = torch.randn(512, 512, device=device)
        c = a @ b  # matmul on GPU
        self.assertEqual(c.device.type, "cuda")
        self.assertEqual(c.shape, (512, 512))

    # Small model (2 linear layers) and train for 1 epoch with random data on GPU.
    def test_one_train_step_on_cuda(self):
        device = torch.device("cuda:0")
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)

        x = torch.randn(1024, 32, device=device)
        y = torch.randn(1024, 1, device=device)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        model.train()
        opt.zero_grad()
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        opt.step()

        self.assertGreaterEqual(loss.item(), 0.0)

    # Validate that the CUDA driver returns valid info and PyTorch can query memory.
    def test_device_props_and_memory(self):
        props = torch.cuda.get_device_properties(0)
        _ = torch.cuda.memory_allocated(0)
        _ = torch.cuda.memory_reserved(0)
        self.assertTrue(props.total_memory > 0, "Could not read total GPU memory.")
        # Basic device name check
        self.assertTrue(len(props.name) > 0, "GPU name is empty.")


if __name__ == "__main__":
    unittest.main()

PY = python
TRAIN_SCRIPT = src/train/train_run.py

help:
	@echo "=== Makefile targets ==="
	@echo "make run-png   - Train with PNG dataset"
	@echo "make run-npy-h - Train with NPY RGB+H dataset"
	@echo "make run-npy-s - Train with NPY RGB+S dataset"
	@echo "make run-all   - Run all experiments"
	@echo "make clean     - Remove caches and results"

run-png:
	$(PY) $(TRAIN_SCRIPT) \
		--data_mode png \
		--train_dir .cache/images/processed/rgb_base/train \
		--test_dir .cache/images/processed/rgb_base/test \
		--external_dir .cache/images/processed/rgb_base/external_val \
		--epochs 25 --batch 32 --tag rgb_png

run-npy-h:
	$(PY) $(TRAIN_SCRIPT) \
		--data_mode npy \
		--train_dir_npy .cache/images/processed/rgb_h_npy/train \
		--test_dir_npy .cache/images/processed/rgb_h_npy/test \
		--external_dir_npy .cache/images/processed/rgb_h_npy/external_val \
		--img_size 224 --batch 32 --epochs 25 --lr 0.001 \
		--classes "Benign,Malignant" --tag rgb_h

run-npy-s:
	$(PY) $(TRAIN_SCRIPT) \
		--data_mode npy \
		--train_dir_npy .cache/images/processed/rgb_s_npy/train \
		--test_dir_npy .cache/images/processed/rgb_s_npy/test \
		--external_dir_npy .cache/images/processed/rgb_s_npy/external_val \
		--img_size 224 --batch 32 --epochs 25 --lr 0.001 \
		--classes "Benign,Malignant" --tag rgb_s

run-all: run-png run-npy-h run-npy-s

clean:
	@echo "Cleaning..."
	-find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	-rm -rf .pytest_cache .ruff_cache .mypy_cache results/runs 2>/dev/null || true

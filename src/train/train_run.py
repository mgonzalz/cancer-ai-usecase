from __future__ import annotations
import argparse, os, json, time
from pathlib import Path
from typing import List
import numpy as np
import tensorflow as tf

from src.train.metrics import (
    save_history_plots,
    plot_roc_auc,
    plot_confusion_matrix,
)
from src.models.cnn_baseline import build_cnn


# ---------------- Utils ----------------
def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------- Data pipelines ----------------
def ds_from_directory_png(
    root_dir: str,
    classes: List[str],
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
):
    """
    Create a PNG/JPG dataset normalized to [0,1], label_mode=categorical (one-hot).
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        root_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=classes,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    # Normalize to [0,1]
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def ds_from_directory_png_eval(root_dir: str, classes: List[str], image_size=(224, 224)):
    """
    EVAL dataset (batch=None, no shuffle) to mirror your evaluation pattern.
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        root_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=classes,
        image_size=image_size,
        batch_size=None,
        shuffle=False,
    )
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def ds_from_npy(root_dir: str, classes: List[str], batch_size=32, shuffle=True):
    """
    Read materialized .npy tensors (3 or 4 channels) and return (x,y).
    Expected structure:
      root_dir/<class>/*.npy
    """
    files = []
    labels = []
    for i, c in enumerate(classes):
        cdir = Path(root_dir) / c
        files_c = sorted(str(p) for p in cdir.rglob("*.npy"))
        files += files_c
        labels += [i] * len(files_c)

    files = np.array(files)
    labels = np.array(labels, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(files), reshuffle_each_iteration=True)

    def _load(path, label):
        def py_load(p):
            import numpy as np
            x = np.load(p.decode("utf-8")).astype("float32")  # [H,W,3 or 4] in [0,1]
            return x

        x = tf.numpy_function(py_load, [path], tf.float32)
        x.set_shape([None, None, None])  # flexible HWC
        y = tf.one_hot(label, depth=len(classes))
        return x, y

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if batch_size:
        ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)


def ds_from_npy_eval(root_dir: str, classes: List[str]):
    """
    Eval dataset with batch=None (yield one item per step), no shuffle.
    """
    files = []
    labels = []
    for i, c in enumerate(classes):
        cdir = Path(root_dir) / c
        files_c = sorted(str(p) for p in cdir.rglob("*.npy"))
        files += files_c
        labels += [i] * len(files_c)

    ds = tf.data.Dataset.from_tensor_slices((files, labels))

    def _load(path, label):
        def py_load(p):
            import numpy as np
            x = np.load(p.decode("utf-8")).astype("float32")
            return x

        x = tf.numpy_function(py_load, [path], tf.float32)
        x.set_shape([None, None, None])
        y = tf.one_hot(label, depth=len(classes))
        return x, y

    # batch=None (one element per step), no shuffle
    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)


# ---------------- Train & Eval ----------------
def train_and_eval(args):
    out_root = ensure_dir(Path("./results/runs") / f"{now_tag()}__{args.tag}")
    curves_dir = ensure_dir(out_root / "curves")
    conf_dir = ensure_dir(out_root / "confusion")
    models_dir = ensure_dir(out_root / "models")

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    K = len(classes)
    img_size = (args.img_size, args.img_size)

    # ---------- Datasets ----------
    if args.data_mode == "png":
        train_ds = ds_from_directory_png(
            args.train_dir, classes, image_size=img_size, batch_size=args.batch, shuffle=True
        )
        test_ds = ds_from_directory_png_eval(args.test_dir, classes, image_size=img_size)
        ext_ds = (
            ds_from_directory_png_eval(args.external_dir, classes, image_size=img_size)
            if args.external_dir
            else None
        )
        input_channels = 3
    else:
        # NPY (4ch possible)
        train_ds = ds_from_npy(Path(args.train_dir_npy), classes, batch_size=args.batch, shuffle=True)
        test_ds = ds_from_npy_eval(Path(args.test_dir_npy), classes)
        ext_ds = ds_from_npy_eval(Path(args.external_dir_npy), classes) if args.external_dir_npy else None
        # Infer channels from first sample
        sample = next(iter(train_ds.unbatch().take(1)))[0].numpy()
        input_channels = sample.shape[-1]

    # Prefetch and (if PNG) batch test for faster eval
    test_ds_batched = test_ds.batch(args.batch).prefetch(tf.data.AUTOTUNE)
    ext_ds_batched = ext_ds.batch(args.batch).prefetch(tf.data.AUTOTUNE) if ext_ds is not None else None

    # ---------- Model ----------
    model = build_cnn((args.img_size, args.img_size, input_channels), K)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="categorical_crossentropy", metrics=["accuracy"])

    # ---------- Callbacks ----------
    # Always monitor validation to pick the best checkpoint and adjust LR.
    ckpt_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(models_dir / "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,  # stop if no improvement
        restore_best_weights=True,
    )
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=6  # reduce LR if no improvement within 'patience' epochs
    )

    # ---------- Train ----------
    # We use the "test" set as validation for logging val_loss/val_accuracy, to plot train vs val curves.
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=test_ds_batched,
        callbacks=[ckpt_best, es, rlrop],
        verbose=1,
    )
    model.save(models_dir / "last_model.keras")
    save_history_plots(history, curves_dir)

    # ---------- Evaluate on TEST ----------
    y_true, y_prob = [], []
    for x, y in test_ds_batched:
        p = model.predict(x, verbose=0)
        # prob of class 1 (Malignant) — assumes binary classes=["Benign","Malignant"]
        y_prob.extend(p[:, 1].tolist())
        y_true.extend(tf.argmax(y, axis=1).numpy().tolist())
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)
    y_pred = (y_prob >= 0.5).astype(int)

    auc_test = plot_roc_auc(y_true, y_prob, curves_dir / "roc_test.png", title="ROC (test)")
    cm_test = plot_confusion_matrix(y_true, y_pred, classes, conf_dir / "test.png", title="Confusion (test)")
    loss_test, acc_test = model.evaluate(test_ds_batched, verbose=0)

    # ---------- Evaluate on EXTERNAL (optional) ----------
    auc_ext = None
    cm_ext = None
    acc_ext = None
    loss_ext = None
    if ext_ds_batched is not None:
        y_true_e, y_prob_e = [], []
        for x, y in ext_ds_batched:
            p = model.predict(x, verbose=0)
            y_prob_e.extend(p[:, 1].tolist())
            y_true_e.extend(tf.argmax(y, axis=1).numpy().tolist())
        y_prob_e = np.array(y_prob_e)
        y_true_e = np.array(y_true_e)
        y_pred_e = (y_prob_e >= 0.5).astype(int)

        auc_ext = plot_roc_auc(y_true_e, y_prob_e, curves_dir / "roc_external.png", title="ROC (external)")
        cm_ext = plot_confusion_matrix(y_true_e, y_pred_e, classes, conf_dir / "external.png", title="Confusion (external)")
        loss_ext, acc_ext = model.evaluate(ext_ds_batched, verbose=0)

    # ---------- Save metrics JSON ----------
    metrics = {
        "train": {"epochs": len(history.history["loss"])},
        "test": {"loss": float(loss_test), "accuracy": float(acc_test), "auc": float(auc_test)},
        "external": {
            "loss": float(loss_ext) if loss_ext is not None else None,
            "accuracy": float(acc_ext) if acc_ext is not None else None,
            "auc": float(auc_ext) if auc_ext is not None else None,
        },
        "classes": classes,
    }
    with open(out_root / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== DONE ===")
    print("Run dir:", out_root)
    print("Best model:", models_dir / "best_model.keras")
    print("Last model:", models_dir / "last_model.keras")
    print("Curves:", curves_dir)
    print("Confusion:", conf_dir)


# ---------------- Main ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Data mode
    ap.add_argument(
        "--data_mode",
        choices=["png", "npy"],
        default="png",
        help="png: image_dataset_from_directory; npy: materialized tensors (3/4 channels).",
    )

    # PNG dirs
    ap.add_argument(
        "--train_dir",
        type=str,
        default="./.cache/images/processed/rgb_base/train",
        help="Root with class subfolders (PNG).",
    )
    ap.add_argument(
        "--test_dir",
        type=str,
        default="./.cache/images/processed/rgb_base/test",
        help="Root with class subfolders (PNG).",
    )
    ap.add_argument(
        "--external_dir",
        type=str,
        default="./data/external_val",
        help="Root with class subfolders (PNG). Optional.",
    )

    # NPY dirs (when using RGB+H / RGB+S)
    ap.add_argument("--train_dir_npy", type=str, default="./.cache/images/processed/rgb_h_npy/train")
    ap.add_argument("--test_dir_npy", type=str, default="./.cache/images/processed/rgb_h_npy/test")
    ap.add_argument("--external_dir_npy", type=str, default="./.cache/images/processed/rgb_h_npy/external_val")

    # Hyperparameters
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--classes", type=str, default="Benign,Malignant")
    ap.add_argument("--tag", type=str, default="baseline")

    # Other
    ap.add_argument(
        "--use_val_from_train",
        action="store_true",
        help="(No effect here) Kept for compatibility; we always use val=test now.",
    )

    # Reduce TF verbosity if desired
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    args = ap.parse_args()
    train_and_eval(args)

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Sequence

__all__ = [
    "save_history_plots",
    "plot_roc_auc",
    "plot_confusion_matrix",
]

def save_history_plots(history, outdir: Path) -> None:
    """
    Save loss/accuracy curves from a tf.keras.callbacks.History object.
    """
    hist = history.history

    # Loss
    plt.figure()
    plt.plot(hist["loss"], label="train")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "loss.png", dpi=130)
    plt.close()

    # Accuracy
    if "accuracy" in hist:
        plt.figure()
        plt.plot(hist["accuracy"], label="train")
        if "val_accuracy" in hist:
            plt.plot(hist["val_accuracy"], label="val")
        plt.title("Accuracy")
        plt.xlabel("epoch")
        plt.ylabel("acc")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "accuracy.png", dpi=130)
        plt.close()


def plot_roc_auc(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    outpath: Path,
    title: str = "ROC (test)",
) -> float:
    """
    Plot ROC curve and compute AUC (no sklearn dependency).

    Args:
        - y_true: iterable of ints (0/1), shape [N]
        - y_prob: iterable of floats in [0,1], shape [N] (positive-class scores)
        - outpath: where to save the plot
        - title: plot title
    Returns:
        - auc (float)
    """
    y_true = np.asarray(y_true).astype(np.int32)
    y_prob = np.asarray(y_prob).astype(np.float32)

    # Sort by descending score
    order = np.argsort(-y_prob)
    y = y_true[order]
    scores = y_prob[order]

    P = (y == 1).sum()
    N = (y == 0).sum()

    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0
    prev_s = None

    for yi, si in zip(y, scores):
        if prev_s is None or si != prev_s:
            tpr.append(tp / (P + 1e-9))
            fpr.append(fp / (N + 1e-9))
            prev_s = si
        if yi == 1:
            tp += 1
        else:
            fp += 1

    tpr.append(1.0)
    fpr.append(1.0)

    # Trapezoidal rule
    auc = 0.0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()

    return float(auc)


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    outpath: Path,
    title: str = "Confusion",
):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    K = len(class_names)
    cm = np.zeros((K, K), dtype=np.int32)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xticks(range(K), class_names, rotation=0, ha="right")
    plt.yticks(range(K), class_names)

    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", color="black")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()

    return cm

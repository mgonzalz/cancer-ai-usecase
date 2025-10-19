# Azure Blob Manifest Builder for Melanoma Dataset
# Purpose: This script connects to an Azure Blob Storage container and automatically
# generates manifest CSV files (train/test) for a melanoma image dataset.
# Each manifest row contains:
# - a signed blob URL (for reproducible access),
# - the inferred class label ("Benign" or "Malignant"),
# - an MD5 hash for file integrity,
# - image dimensions (width, height),
# - and the dataset split ("train" or "test").
#
# Notes:
# - It assumes images are organized in Azure like:
#   raw/automatic-learning/melanoma/train/<class>/*.jpg
#   raw/automatic-learning/melanoma/test/<class>/*.jpg
# - Labels are inferred from folder names containing "benign" or "malignant".
# - Requires valid Azure account URL and SAS token in config/base.yaml.
# - CSVs are saved under the folder specified in cfg["paths"]["manifests_dir"].


from __future__ import annotations

import argparse
import hashlib
import io
import pathlib
from datetime import datetime
from typing import Iterator, Tuple

import pandas as pd
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from src.utils.config import load_config

# Common aliases for label inference (typos or language variations)
BENIGN_ALIASES = {"benign", "benigno"}
MALIGNANT_ALIASES = {"malignant", "maligno"}


# Helper functions
def infer_label_from_blob_name(blob_name: str) -> str:
    """
    Infers the label ("Benign" or "Malignant") from an Azure blob path.
    Args:
        - blob_name (str): Full blob name or path in Azure Storage.
            - Example: "automatic-learning/melanoma/train/benign/img1.jpg"
    Returns:
        - str: The inferred class label ("Benign" or "Malignant").
    Raises:
        ValueError: If no known label keywords are found in the path.
    """
    lower = blob_name.lower()
    parts = [p for p in lower.split("/") if p]
    if any(p in BENIGN_ALIASES for p in parts):
        return "Benign"
    if any(p in MALIGNANT_ALIASES for p in parts):
        return "Malignant"
    raise ValueError(f"No se pudo inferir label desde la ruta: {blob_name}")


def compute_md5_and_dims(content: bytes) -> Tuple[str, Tuple[int, int]]:
    """
    Computes the MD5 checksum and (width, height) dimensions of an image.
    Args:
        - content (bytes): Raw image bytes.
    Returns:
        - Tuple[str, Tuple[int, int]]: A tuple containing:
            - md5 (str): Hexadecimal MD5 hash string.
            - (w, h) (Tuple[int, int]): Image dimensions (width, height).
    """
    md5 = hashlib.md5(content).hexdigest()
    with Image.open(io.BytesIO(content)) as img:
        w, h = img.size
    return md5, (w, h)


def iter_blobs(
    service: BlobServiceClient, container: str, prefix: str
) -> Iterator[str]:
    """
    Iterates over all image blobs under a given prefix in a specific container.
    Args:
        - service (BlobServiceClient): Authenticated Azure Blob Service client.
        - container (str): Name of the Azure container.
        - prefix (str): Path prefix to search for blobs.

    Yields:
        - str: The blob name for each image file found (.jpg, .jpeg, .png, .bmp).
    """
    container_client = service.get_container_client(container)
    # List only blobs (omit virtual directories)
    for blob in container_client.list_blobs(name_starts_with=prefix):
        # Filter only common image formats
        if blob.name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            yield blob.name


def download_blob_bytes(
    service: BlobServiceClient, container: str, blob_name: str
) -> bytes:
    """
    Downloads the full content of a blob as bytes.
    Args:
        - service (BlobServiceClient): Authenticated Azure Blob Service client.
        - container (str): Azure container name.
        - blob_name (str): Name of the blob to download.
    Returns:
        - bytes: Raw content of the blob.
    """
    bc = service.get_blob_client(container=container, blob=blob_name)
    stream = bc.download_blob()
    return stream.readall()


# Manifest builder
def build_manifest_for_split(cfg: dict, split: str) -> pd.DataFrame:
    """
    Builds a manifest DataFrame for a dataset split.
    It lists all image blobs, infers their labels, computes hashes/dimensions,
    and stores this metadata in a pandas DataFrame.
    Args:
        - cfg (dict): Configuration dictionary loaded from YAML (contains Azure info).
        - split (str): Either "train" or "test".

    Returns:
        pd.DataFrame: A DataFrame containing columns:
            - path: Signed blob URL for reproducible access.
            - label: Inferred class label ("Benign" / "Malignant").
            - hash: MD5 checksum of the image.
            - width: Image width in pixels.
            - height: Image height in pixels.
            - split: Dataset split name ("train" or "test").
    """
    account_url = cfg["azure"]["account_url"]
    container = cfg["azure"]["container_raw"]
    sas = cfg["azure"]["sas_token"]
    # Example Azure prefix: "automatic-learning/melanoma/train"
    azure_prefix = f"automatic-learning/melanoma/{split}"

    # Connect to Azure Blob Storage
    service = BlobServiceClient(account_url=f"{account_url}{sas}")
    rows = []
    print(f"[INFO] Listing blobs in {container}/{azure_prefix} ...")
    blob_names = list(iter_blobs(service, container, azure_prefix))

    # Process all image blobs
    for name in tqdm(blob_names, desc=f"Probing {split}", unit="img"):
        label = infer_label_from_blob_name(name)
        url = f"{name}"  # Use relative path; signed URL can be constructed as needed
        content = download_blob_bytes(service, container, name)
        md5, (w, h) = compute_md5_and_dims(content)
        rows.append(
            {
                "path": url,
                "label": label,
                "hash": md5,
                "width": w,
                "height": h,
                "split": split,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[["path", "label", "hash", "width", "height", "split"]]
    return df


def main():
    """
    Command-line entry point.
    Reads configuration, connects to Azure, builds manifests for both
    'train' and 'test' splits, and saves them as CSV files.
    Command:
        - python src/dataio/manifests.py --cfg config/base.yaml
    """
    parser = argparse.ArgumentParser(
        description="Generates train/test manifests from Azure Blob."
    )
    parser.add_argument(
        "--cfg", default="config/base.yaml", help="Path to config/base.yaml"
    )
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.cfg)
    print(f"[INFO] Loaded config from: {args.cfg}")
    print(cfg)

    # Ensure output directory exists
    out_dir = pathlib.Path(cfg["paths"]["manifests_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build manifests for both splits
    df_train = build_manifest_for_split(cfg, "train")
    df_test = build_manifest_for_split(cfg, "test")

    # Save to CSV
    train_csv = out_dir / "train_manifest.csv"
    test_csv = out_dir / "test_manifest.csv"

    df_train.to_csv(train_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    print(f"[OK] Saved: {train_csv} ({len(df_train)} rows)")
    print(f"[OK] Saved: {test_csv} ({len(df_test)} rows)")

    # Quick log
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[INFO] Manifest run {ts} completed successfully.")


if __name__ == "__main__":
    main()

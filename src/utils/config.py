# YAML configuration loader & validator
# Purpose:
# - Load a YAML configuration file for the project (e.g., config/base.yaml).
# - Validate that the configuration contains the required sections/keys.
# - Raise clear exceptions (FileNotFoundError, ConfigError) when something is missing.
#
# Notes:
# - Expected top-level keys: seed, img_size, batch_size, epochs, learning_rate, azure, paths.
# - Expected azure keys: account_url, container_raw, sas_token.
# - Expected paths keys: manifests_dir, results_dir, cache_dir.

from __future__ import annotations

import os
import pathlib
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigError(Exception):
    """Custom exception for configuration errors (missing or mistyped keys)."""


def _assert_keys(name: str, data: dict, required: set[str]) -> None:
    """
    Validates that a given dictionary contains all required keys.
    Raises a ConfigError if any expected key is missing.
    """
    missing = required - set(data.keys())
    if missing:
        raise ConfigError(f"[{name}] missing required keys: {sorted(missing)}")


def load_config(cfg_path: str) -> dict[str, Any]:
    """
    Loads a YAML configuration file, expands environment variables (${VAR})
    defined in a .env file, and validates the expected structure.

    Args:
        - cfg_path: Path to the YAML configuration file.

    Returns:
        - A dictionary containing the parsed configuration data.
    """
    load_dotenv()  # Load environment variables from .env file (if present)

    path = pathlib.Path(cfg_path)  # Convert string path to Path object
    if not path.exists():
        raise FileNotFoundError(cfg_path)

    # Read the raw YAML file content
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Replace environment variable placeholders with actual values
    for key, val in os.environ.items():
        raw = raw.replace(f"${{{key}}}", val)

    # Parse YAML safely into a Python dictionary
    data = yaml.safe_load(raw)

    # Validate the main configuration keys
    _assert_keys(
        "root",
        data,
        {"seed", "img_size", "batch_size", "epochs", "learning_rate", "azure", "paths"},
    )
    _assert_keys("azure", data["azure"], {"account_url", "container_raw", "sas_token"})
    _assert_keys("paths", data["paths"], {"manifests_dir", "results_dir", "cache_dir"})

    return data

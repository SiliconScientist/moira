from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from moira.config_base import load_toml_file
from moira.mlip_config import IngestConfig, MLIPConfig


class Config(BaseModel):
    ingest: IngestConfig
    mlip: MLIPConfig


def get_config(path: str | Path = "config.toml") -> Config:
    return Config.model_validate(load_toml_file(Path(path)))

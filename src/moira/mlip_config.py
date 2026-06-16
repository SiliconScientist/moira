from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class StoichConfig(BaseModel):
    elements: list[str]
    basis_species: list[str]
    basis_composition: Dict[str, Dict[str, int]]


class IngestConfig(BaseModel):
    source: Path
    dataset_name: str
    profile: str = "vasp_mapping"
    catbench_folder: Optional[Path] = None
    stoich: StoichConfig


class MLIPModelsConfig(BaseModel):
    enabled: List[str]


class RootstockModelConfig(BaseModel):
    model: str
    mlip_name: str
    checkpoint: Optional[str] = None
    output_model: Optional[str] = None
    model_version: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class RootstockConfig(BaseModel):
    root: Path
    python: Optional[Path] = None
    models: Dict[str, RootstockModelConfig]


class MLIPConfig(BaseModel):
    dev_n: int
    dev_run: bool
    dataset: Optional[str] = None
    datasets: Optional[List[str]] = None
    results_dir: Optional[Path] = None
    adapter_backend: str = "rootstock"
    optimizer: str = "LBFGS"
    shard_size: Optional[int] = Field(default=None, ge=1)
    num_shards: Optional[int] = Field(default=None, ge=1)
    shard_index: Optional[int] = Field(default=None, ge=0)
    models: MLIPModelsConfig
    rootstock: RootstockConfig

    @model_validator(mode="after")
    def validate_sharding(self) -> "MLIPConfig":
        if self.shard_size is not None and self.num_shards is not None:
            raise ValueError("Configure only one of mlip.shard_size or mlip.num_shards")
        if self.shard_index is not None and self.shard_size is None and self.num_shards is None:
            raise ValueError(
                "mlip.shard_index requires either mlip.shard_size or mlip.num_shards"
            )
        if self.shard_index is not None and self.num_shards is not None and self.shard_index >= self.num_shards:
            raise ValueError("mlip.shard_index must be smaller than mlip.num_shards")
        return self


def raw_dataset_path(raw_dataset_filename: str) -> Path:
    return Path("data/raw_data") / raw_dataset_filename


def mlip_results_dir(mlip_run_dirname: str) -> Path:
    return Path("data/mlips") / mlip_run_dirname


def default_catbench_folder(source: Path) -> Path:
    return source.parent / f"{source.name}_catbench"


def fill_mlip_dataset_path(
    mlip_config: MLIPConfig,
    *,
    dataset_path: Path | None,
) -> None:
    if mlip_config.dataset is None and dataset_path is not None:
        mlip_config.dataset = str(dataset_path)

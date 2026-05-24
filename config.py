from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CLIPConfig:
    model_name: str = "ViT-L/14"
    library: str = "openai"
    local_model_dir: str = "/your/pth/pretrained"
    preprocess_type: str = "targetpad"
    targetpad_ratio: float = 1.25
    use_local: bool = True


@dataclass(frozen=True)
class CalibrationConfig:
    alpha: float = 0.9
    beta: float = 0.8
    lambda_weight: float = 0.5
    saved_topk: int = 50
    internal_topk: int = 256
    batch_size: int = 64
    log_every: int = 25


@dataclass(frozen=True)
class VerificationConfig:
    rho: float = 0.4
    topk: int = 50
    image_batch_size: int = 256
    text_batch_size: int = 512
    log_every: int = 50


@dataclass(frozen=True)
class MetricConfig:
    fashioniq_recall_cutoffs: Tuple[int, ...] = (1, 5, 10, 50)
    cirr_recall_cutoffs: Tuple[int, ...] = (1, 5, 10, 50)
    cirr_group_recall_cutoffs: Tuple[int, ...] = (1, 2, 3)
    circo_cutoffs: Tuple[int, ...] = (5, 10, 25, 50)
    circo_ranking_depth: int = 50


@dataclass(frozen=True)
class DatasetConfig:
    default_split: str = "test"
    fashioniq_types: Tuple[str, ...] = ("dress", "shirt", "toptee")


@dataclass(frozen=True)
class NumericalConfig:
    eps: float = 1e-6
    slerp_linear_threshold: float = 0.9995


@dataclass(frozen=True)
class CoVerConfig:
    clip: CLIPConfig = CLIPConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    verification: VerificationConfig = VerificationConfig()
    metrics: MetricConfig = MetricConfig()
    dataset: DatasetConfig = DatasetConfig()
    numerical: NumericalConfig = NumericalConfig()


DEFAULT_CONFIG = CoVerConfig()

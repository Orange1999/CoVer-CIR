from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

from clip_loader import CLIPLoader
from config import DEFAULT_CONFIG


ImageInput = Union[str, Path, Image.Image, torch.Tensor]
TextInput = Union[str, Sequence[str]]


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=-1)


def spherical_interpolation(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    gamma: float,
    eps: float = DEFAULT_CONFIG.numerical.eps,
) -> torch.Tensor:
    image_features = normalize_features(image_features)
    text_features = normalize_features(text_features)
    dot = (image_features * text_features).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta).clamp_min(eps)
    scale_image = torch.sin((1.0 - gamma) * theta) / sin_theta
    scale_text = torch.sin(gamma * theta) / sin_theta
    slerp_features = scale_image * image_features + scale_text * text_features
    linear_features = image_features + gamma * (text_features - image_features)
    close = torch.abs(dot) > DEFAULT_CONFIG.numerical.slerp_linear_threshold
    return normalize_features(torch.where(close, linear_features, slerp_features))


class TargetPad:
    def __init__(self, target_ratio: float, size: int):
        self.target_ratio = target_ratio
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        actual_ratio = max(width, height) / min(width, height)
        if actual_ratio < self.target_ratio:
            return image
        scaled_max_wh = max(width, height) / self.target_ratio
        horizontal_pad = max(int((scaled_max_wh - width) / 2), 0)
        vertical_pad = max(int((scaled_max_wh - height) / 2), 0)
        return TF.pad(image, [horizontal_pad, vertical_pad, horizontal_pad, vertical_pad], 0, "constant")


def targetpad_transform(target_ratio: float, size: int):
    return Compose(
        [
            TargetPad(target_ratio, size),
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )


class CoVerCLIPFeatureExtractor:
    def __init__(
        self,
        model_name: str = DEFAULT_CONFIG.clip.model_name,
        library: str = DEFAULT_CONFIG.clip.library,
        device: Optional[Union[str, torch.device, int]] = None,
        local_model_dir: Optional[str] = DEFAULT_CONFIG.clip.local_model_dir,
        pretrained: Optional[str] = None,
        use_local: bool = DEFAULT_CONFIG.clip.use_local,
        preprocess_type: str = DEFAULT_CONFIG.clip.preprocess_type,
        targetpad_ratio: float = DEFAULT_CONFIG.clip.targetpad_ratio,
        normalize: bool = True,
    ):
        self.device = self._resolve_device(device)
        self.normalize = normalize
        self.loader = CLIPLoader(
            library=library,
            device=self.device,
            local_model_dir=local_model_dir,
        )
        self.model, self.clip_preprocess = self.loader.load(
            model_name,
            pretrained=pretrained,
            use_local=use_local,
        )
        self.model = self.model.eval().requires_grad_(False)
        self.preprocess = self._build_preprocess(preprocess_type, targetpad_ratio)

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device, int]]) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, int):
            return torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _build_preprocess(self, preprocess_type: str, targetpad_ratio: float):
        if preprocess_type == "clip":
            return self.clip_preprocess
        if preprocess_type == "targetpad":
            first_transform = self.clip_preprocess.transforms[0]
            size = first_transform.size
            if isinstance(size, (tuple, list)):
                size = size[0]
            return targetpad_transform(targetpad_ratio, size)
        raise ValueError("preprocess_type must be 'clip' or 'targetpad'")

    def preprocess_image(self, image: ImageInput) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            return image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")
        return self.preprocess(image)

    def _batch_images(self, images: Union[ImageInput, Sequence[ImageInput]]) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            return images.unsqueeze(0) if images.ndim == 3 else images
        if isinstance(images, (str, Path, Image.Image)):
            return self.preprocess_image(images).unsqueeze(0)
        return torch.stack([self.preprocess_image(image) for image in images], dim=0)

    @torch.no_grad()
    def extract_image_features(
        self,
        images: Union[ImageInput, Sequence[ImageInput]],
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
    ) -> torch.Tensor:
        image_batch = self._batch_images(images)
        normalize = self.normalize if normalize is None else normalize
        if batch_size is None or image_batch.shape[0] <= batch_size:
            features = self.loader.extract_image_features(self.model, image_batch.to(self.device))
            return normalize_features(features) if normalize else features.float()
        chunks = []
        for start in range(0, image_batch.shape[0], batch_size):
            batch = image_batch[start : start + batch_size].to(self.device)
            features = self.loader.extract_image_features(self.model, batch)
            chunks.append((normalize_features(features) if normalize else features.float()).cpu())
        return torch.cat(chunks, dim=0).to(self.device)

    @torch.no_grad()
    def extract_text_features(
        self,
        texts: TextInput,
        batch_size: Optional[int] = None,
        truncate: bool = True,
        normalize: Optional[bool] = None,
    ) -> torch.Tensor:
        if isinstance(texts, str):
            text_list = [texts]
        else:
            text_list = list(texts)
        normalize = self.normalize if normalize is None else normalize
        if batch_size is None or len(text_list) <= batch_size:
            features = self.loader.extract_text_features(self.model, text_list, truncate=truncate)
            return normalize_features(features) if normalize else features.float()
        chunks = []
        for start in range(0, len(text_list), batch_size):
            batch = text_list[start : start + batch_size]
            features = self.loader.extract_text_features(self.model, batch, truncate=truncate)
            chunks.append((normalize_features(features) if normalize else features.float()).cpu())
        return torch.cat(chunks, dim=0).to(self.device)

    @torch.no_grad()
    def compose_image_text_features(
        self,
        images: Union[ImageInput, Sequence[ImageInput], torch.Tensor],
        texts: TextInput,
        gamma: float,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        image_features = images if isinstance(images, torch.Tensor) and images.ndim == 2 else self.extract_image_features(images, batch_size=batch_size)
        text_features = self.extract_text_features(texts, batch_size=batch_size)
        image_features, text_features = self._broadcast_features(image_features, text_features)
        return spherical_interpolation(image_features, text_features, gamma)

    @staticmethod
    def _broadcast_features(image_features: torch.Tensor, text_features: torch.Tensor):
        if image_features.shape[0] == text_features.shape[0]:
            return image_features, text_features
        if image_features.shape[0] == 1:
            return image_features.expand(text_features.shape[0], -1), text_features
        if text_features.shape[0] == 1:
            return image_features, text_features.expand(image_features.shape[0], -1)
        raise ValueError(
            "Image and text batch sizes must match, or one side must contain a single item."
        )

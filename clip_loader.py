from pathlib import Path
from typing import List, Optional, Union

import torch


class CLIPLoader:
    OPENAI_CLIP_MODELS = {
        "RN50",
        "RN101",
        "RN50x4",
        "RN50x16",
        "RN50x64",
        "ViT-B/32",
        "ViT-B/16",
        "ViT-L/14",
        "ViT-L/14@336px",
    }

    OPENCLIP_MODELS = {
        "ViT-B-32": "CLIP-ViT-B-32-laion2B-s34B-b79K",
        "ViT-B-16": "CLIP-ViT-B-16-laion2B-s34B-b88K",
        "ViT-L-14": "CLIP-ViT-L-14-laion2B-s32B-b82K",
        "ViT-H-14": "CLIP-ViT-H-14-laion2B-s32B-b79K",
        "ViT-g-14": "laion_CLIP-ViT-g-14-laion2B-s34B-b88K",
        "ViT-bigG-14": "CLIP-ViT-bigG-14-laion2B-39B-b160k",
    }

    def __init__(self, library: str = "openai", device=None, local_model_dir: Optional[str] = None):
        self.library = library.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.local_model_dir = local_model_dir
        self.local_checkpoint_map = {}
        if self.local_model_dir and self.library == "open_clip":
            self._build_local_checkpoint_map()

    def _build_local_checkpoint_map(self) -> None:
        local_dir = Path(self.local_model_dir)
        if not local_dir.exists():
            return
        for model_name, folder_name in self.OPENCLIP_MODELS.items():
            checkpoint_file = local_dir / folder_name / "open_clip_pytorch_model.bin"
            if checkpoint_file.exists():
                self.local_checkpoint_map[model_name] = str(checkpoint_file)

    def load(self, model_name: str, pretrained: Optional[str] = None, use_local: bool = True):
        if self.library == "openai":
            import clip

            if model_name not in self.OPENAI_CLIP_MODELS:
                raise ValueError(f"Unsupported OpenAI CLIP model: {model_name}")
            return clip.load(model_name, device=self.device)

        if self.library == "open_clip":
            import open_clip

            if model_name not in open_clip.list_models():
                raise ValueError(f"Unsupported OpenCLIP model: {model_name}")
            final_pretrained = pretrained
            if pretrained and Path(pretrained).exists():
                final_pretrained = pretrained
            elif use_local and pretrained is None:
                final_pretrained = self.local_checkpoint_map.get(model_name)
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=final_pretrained,
                device=self.device,
            )
            model.tokenizer = open_clip.get_tokenizer(model_name)
            return model, preprocess

        raise ValueError("library must be 'openai' or 'open_clip'")

    def extract_image_features(self, model, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        with torch.no_grad():
            if str(self.device).startswith("cuda"):
                with torch.cuda.amp.autocast():
                    return model.encode_image(images)
            return model.encode_image(images)

    def extract_text_features(self, model, text_input: Union[str, List[str]], truncate: bool = False) -> torch.Tensor:
        if isinstance(text_input, str):
            text_input = [text_input]
        if self.library == "open_clip":
            tokens = model.tokenizer(text_input)
        else:
            import clip

            tokens = clip.tokenize(text_input, context_length=77, truncate=truncate)
        tokens = tokens.to(self.device)
        with torch.no_grad():
            if str(self.device).startswith("cuda"):
                with torch.cuda.amp.autocast():
                    return model.encode_text(tokens)
            return model.encode_text(tokens)

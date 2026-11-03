"""Reusable CPU image classification without HTTP or import-time model loading."""

from contextlib import closing
from pathlib import Path

import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessorPil


MODEL_ID = "google/vit-base-patch16-224"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"


class ImageClassifier:
    """Load one checkpoint on CPU at construction and reuse it for predictions."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.processor = ViTImageProcessorPil.from_pretrained(
            MODEL_ID, cache_dir=self.cache_dir
        )
        self.model = ViTForImageClassification.from_pretrained(
            MODEL_ID, cache_dir=self.cache_dir
        )
        self.model.to("cpu")
        self.model.eval()

    def classify(self, image: Image.Image) -> dict[str, list[dict[str, str | float]]]:
        """Return three ranked labels/scores for a Pillow image without mutating it.

        Transparency is composited onto white, then pixels are converted to RGB.
        The checkpoint processor handles resizing, rescaling, and normalization.
        File-format, upload-size, and other HTTP checks belong outside this service.
        Decode, preprocessing, and inference exceptions propagate to the caller.
        """
        if "A" in image.getbands() or "transparency" in image.info:
            with closing(image.convert("RGBA")) as rgba:
                with closing(Image.new("RGBA", rgba.size, (255, 255, 255, 255))) as background:
                    with closing(Image.alpha_composite(background, rgba)) as composite:
                        rgb = composite.convert("RGB")
        else:
            rgb = image.convert("RGB")

        with closing(rgb):
            inputs = self.processor(images=rgb, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
            scores, indices = torch.topk(probabilities[0], k=3, sorted=True)

        return {
            "predictions": [
                {"label": self.model.config.id2label[index], "score": float(score)}
                for index, score in zip(indices.tolist(), scores.tolist())
            ]
        }

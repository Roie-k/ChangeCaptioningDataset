import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ALL_CLASSES: List[str] = [
    "tree",
    "building",
    "road_paved",
    "parking_lot",
    "driveway",
    "water",
    "sidewalk",
    "athletic_field",
    "trail",
    "railway_tracks",
    "road_unpaved",
    "swimming_pool",
    "crosswalk",
    "painted_median",
    "shipping_container",
    "bike_lane",
    "track",
]

CLASS_PROMPTS: Dict[str, str] = {
    "tree": "an aerial image that contains at least one tree",
    "building": "an aerial image that contains a house or a building",
    "road_paved": "an aerial image that contains a paved road",
    "parking_lot": "an aerial image that contains a parking lot",
    "driveway": "an aerial image that contains a driveway",
    "water": "an aerial image that contains water",
    "sidewalk": "an aerial image that contains a sidewalk",
    "athletic_field": "an aerial image of an athletic field",
    "trail": "an aerial image that contains a trail",
    "railway_tracks": "an aerial image that contains railway tracks",
    "road_unpaved": "an aerial image that contains an unpaved road",
    "swimming_pool": "an aerial image of a swimming pool",
    "crosswalk": "an aerial image of a crosswalk",
    "painted_median": "an aerial image that contains a painted median",
    "shipping_container": "an aerial image that contains a shipping container",
    "bike_lane": "an aerial image of a bike lane",
    "track": "an aerial image that contains a running track",
}


class SemanticScreeningFilter:
    """
    Public semantic screening stage.

    This module performs image-text semantic screening on localized patches
    using a public CLIP backbone. It is a public approximation of the paper's
    semantic filtering stage.

    Screening logic:
    - crop the same region from the before and after images
    - embed each crop
    - compare against class prompts
    - check whether the expected class appears in top-k for each timestamp
    """

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        default_top_k: int = 5,
        min_similarity: Optional[float] = None,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.default_top_k = default_top_k
        self.min_similarity = min_similarity

        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()

        self.class_names: List[str] = list(ALL_CLASSES)
        self.text_prompts: List[str] = [CLASS_PROMPTS[c] for c in self.class_names]
        self.text_features = self._precompute_text_features()

    @torch.no_grad()
    def _precompute_text_features(self) -> torch.Tensor:
        inputs = self.processor(
            text=self.text_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

    @staticmethod
    def _expand_box(
        img_width: int,
        img_height: int,
        xmin: int,
        ymin: int,
        width: int,
        height: int,
        expansion_ratio: float = 0.5,
    ) -> Tuple[int, int, int, int]:
        pad_x = int(np.ceil(width * expansion_ratio))
        pad_y = int(np.ceil(height * expansion_ratio))

        left = max(0, xmin - pad_x)
        top = max(0, ymin - pad_y)
        right = min(img_width, xmin + width + pad_x)
        bottom = min(img_height, ymin + height + pad_y)

        return left, top, right, bottom

    def _crop_patch(
        self,
        image: Image.Image,
        change_rect: Dict[str, int],
    ) -> Optional[Image.Image]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        xmin = int(change_rect["xmin"])
        ymin = int(change_rect["ymin"])
        width = int(change_rect["width"])
        height = int(change_rect["height"])

        left, top, right, bottom = self._expand_box(
            img_width=image.width,
            img_height=image.height,
            xmin=xmin,
            ymin=ymin,
            width=width,
            height=height,
        )

        if right <= left or bottom <= top:
            return None

        return image.crop((left, top, right, bottom))

    @torch.no_grad()
    def _score_patch(self, patch: Image.Image) -> Tuple[List[str], List[float]]:
        inputs = self.processor(images=patch, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        similarities = (image_features @ self.text_features.T).squeeze(0)
        scores = similarities.detach().cpu().tolist()

        ranked = sorted(
            zip(self.class_names, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        ranked_names = [name for name, _ in ranked]
        ranked_scores = [score for _, score in ranked]
        return ranked_names, ranked_scores

    def analyze_change(
        self,
        img1: Image.Image,
        img2: Image.Image,
        change_rect: Dict[str, int],
        change_class_name: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Analyze whether the expected class appears among the top-k predictions
        in the before and after patches.

        Returns a structured dictionary for easier downstream debugging.
        """
        top_k = top_k or self.default_top_k

        if change_class_name not in CLASS_PROMPTS:
            logger.warning("Unknown class for semantic screening: %s", change_class_name)
            return {
                "before_hit": False,
                "after_hit": False,
                "before_top_classes": [],
                "after_top_classes": [],
                "before_score": None,
                "after_score": None,
            }

        patch1 = self._crop_patch(img1, change_rect)
        patch2 = self._crop_patch(img2, change_rect)

        if patch1 is None or patch2 is None:
            return {
                "before_hit": False,
                "after_hit": False,
                "before_top_classes": [],
                "after_top_classes": [],
                "before_score": None,
                "after_score": None,
            }

        before_names, before_scores = self._score_patch(patch1)
        after_names, after_scores = self._score_patch(patch2)

        before_top = before_names[:top_k]
        after_top = after_names[:top_k]

        before_score = None
        after_score = None

        if change_class_name in before_names:
            before_score = before_scores[before_names.index(change_class_name)]
        if change_class_name in after_names:
            after_score = after_scores[after_names.index(change_class_name)]

        before_hit = change_class_name in before_top
        after_hit = change_class_name in after_top

        if self.min_similarity is not None:
            before_hit = before_hit and before_score is not None and before_score >= self.min_similarity
            after_hit = after_hit and after_score is not None and after_score >= self.min_similarity

        result = {
            "before_hit": before_hit,
            "after_hit": after_hit,
            "before_top_classes": before_top,
            "after_top_classes": after_top,
            "before_score": before_score,
            "after_score": after_score,
        }

        logger.info(
            "Semantic screening for class=%s | before_hit=%s after_hit=%s",
            change_class_name,
            before_hit,
            after_hit,
        )
        return result

    def keep_candidate(
        self,
        img1: Image.Image,
        img2: Image.Image,
        change_rect: Dict[str, int],
        change_class_name: str,
        top_k: Optional[int] = None,
    ) -> bool:
        """
        Public boolean decision helper.

        A candidate is kept if the target class appears in the top-k predictions
        for at least one timestamp. Ambiguous cases, such as appearing in both
        timestamps, can then be forwarded to the retrieval-augmented verifier.
        """
        analysis = self.analyze_change(
            img1=img1,
            img2=img2,
            change_rect=change_rect,
            change_class_name=change_class_name,
            top_k=top_k,
        )
        return bool(analysis["before_hit"] or analysis["after_hit"])

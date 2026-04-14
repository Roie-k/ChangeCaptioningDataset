import torch
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Configure logging for transparency and debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Standardized Prompts for External Models ---
# Descriptive prompts help CLIP better understand satellite-view features.
ALL_CLASSES: List[str] = [
    "an aerial image that contains at least one tree",
    "an aerial image that contains a house or a building",
    "an aerial image that contains a paved road",
    "an aerial image that contains a parking lot",
    "an aerial image that contains a driveway",
    "an aerial image that contains water",
    "an aerial image that contains a sidewalk",
    "an aerial image of an athletic field",
    "an aerial image that contains a trail",
    "an aerial image that contains a railway tracks",
    "an aerial image that contains a unpaved road",
    "an image of a swimming pool",
    "an aerial image of a crosswalk",
    "an aerial image that contains a painted median",
    "an aerial image that contains a shipping container",
    "an aerial image of a bike lane",
    "an aerial image that contains a track",
]

SIMPLE_CLASS_NAMES: List[str] = [
    "tree", "building", "road_paved", "parking_lot", "driveway", 
    "water", "sidewalk", "athletic_field", "trail", "railway_tracks", 
    "road_unpaved", "swimming_pool", "crosswalk", "painted_median", 
    "shipping_container", "bike_lane", "track"
]

CLASS_TO_PROMPT_MAP: Dict[str, str] = dict(zip(SIMPLE_CLASS_NAMES, ALL_CLASSES))

class CLIPScreeningFilter:
    """
    Performs semantic verification of detected changes using CLIP.
    
    This class replaces internal SigLIP/FSU filtering to ensure 
    reproducibility using public pre-trained weights.
    """

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        """
        Initializes the CLIP model and processor.
        
        Args:
            model_id: Hugging Face model identifier for the CLIP weights.
        """
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: CLIPModel = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()
        
        # Pre-compute text embeddings for all classes to speed up inference
        self.text_features: torch.Tensor = self._precompute_text_features()

    @torch.no_grad()
    def _precompute_text_features(self) -> torch.Tensor:
        """Encodes the standard class prompts into a normalized feature space."""
        inputs = self.processor(text=ALL_CLASSES, return_tensors="pt", padding=True).to(self.device)
        text_embeds = self.model.get_text_features(**inputs)
        return text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    def analyze_change(
        self,
        img1: Image.Image,
        img2: Image.Image,
        change_rect: Dict[str, int],
        change_class_name: str,
        top_k: int = 5
    ) -> Tuple[bool, bool]:
        """
        Determines if a specific class is present in the Top-K results of both images.
        
        Args:
            img1: PIL image from timestamp 1.
            img2: PIL image from timestamp 2.
            change_rect: Dictionary containing 'xmin', 'ymin', 'width', 'height'.
            change_class_name: The semantic label identified by the segmenter.
            top_k: Sensitivity threshold for validation.
            
        Returns:
            Tuple of (in_top_k_img1, in_top_k_img2).
        """
        logger.info(f"Screening change for class: {change_class_name}")
        
        target_prompt: Optional[str] = CLASS_TO_PROMPT_MAP.get(change_class_name)
        if not target_prompt:
            logger.warning(f"Class {change_class_name} not found in prompt map.")
            return False, False

        # 1. Coordinate Preparation & Expansion
        # We expand the crop slightly to give CLIP contextual information.
        xmin, ymin = change_rect["xmin"], change_rect["ymin"]
        w, h = change_rect["width"], change_rect["height"]
        
        pad_x, pad_y = int(np.ceil(w / 2)), int(np.ceil(h / 2))
        
        # Define crop boundaries
        left = max(0, xmin - pad_x)
        top = max(0, ymin - pad_y)
        right = min(img1.width, xmin + w + pad_x)
        bottom = min(img1.height, ymin + h + pad_y)

        if right <= left or bottom <= top:
            return False, False

        # 2. Extract Patches
        patch1: Image.Image = img1.crop((left, top, right, bottom))
        patch2: Image.Image = img2.crop((left, top, right, bottom))

        # 3. Calculate Similarity
        results: List[bool] = []
        for patch in [patch1, patch2]:
            inputs = self.processor(images=patch, return_tensors="pt").to(self.device)
            image_embeds = self.model.get_image_features(**inputs)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

            # Cosine similarity against pre-computed text
            similarities = (image_embeds @ self.text_features.T).squeeze(0)
            
            # Get indices of Top-K highest scores
            top_indices = torch.topk(similarities, k=min(top_k, len(ALL_CLASSES))).indices
            top_prompts = [ALL_CLASSES[idx] for idx in top_indices.tolist()]
            
            results.append(target_prompt in top_prompts)

        logger.info(f"Screening results for {change_class_name}: T1={results[0]}, T2={results[1]}")
        return results[0], results[1]

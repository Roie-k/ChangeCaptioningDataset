import torch
import numpy as np
import dataclasses
from typing import List, Optional, Tuple, Iterator, Dict, Union
from PIL import Image
from scipy import ndimage
from transformers import (
    AutoImageProcessor, 
    Mask2FormerForUniversalSegmentation, 
    CLIPProcessor, 
    CLIPModel
)

# --- Configuration & Hyperparameters ---
# These thresholds define the "significance" of a detected change.
IOU_THRESH: float = 0.18
MIN_REGION_PIXELS: int = 500
MIN_CHANGED_PIXELS: int = 550
MIN_TOTAL_CHANGED_PIXELS: int = 490

# Standardized Satellite Classes for the mfaytin/mask2former-satellite model
SATELLITE_CLASSES: List[str] = [
    "building", "driveway", "ground_natural", "parking_lot", "road_paved",
    "road_unpaved", "sidewalk", "trail", "tree", "water", "other_human_made",
    "track", "bike_lane", "crosswalk", "painted_median", "railway_tracks",
    "shipping_container", "stairs", "swimming_pool", "athletic_field"
]

@dataclasses.dataclass
class ChangeInstance:
    """
    Metadata and spatial data for a single detected change instance.
    
    Attributes:
        class_name: The human-readable semantic label (e.g., 'building').
        class_id: The integer ID corresponding to SATELLITE_CLASSES.
        footprint: A binary mask (bool) of the instance's total area.
        iou: Intersection over Union of this class between the two timestamps.
        changed_px: Number of pixels that actually flipped value within this footprint.
        bbox: Bounding box in (xmin, ymin, xmax, ymax) format.
        dominant_image_idx: Which image (1 or 2) contains the majority of this class.
    """
    class_name: str
    class_id: int
    footprint: np.ndarray
    iou: float
    changed_px: int
    bbox: Tuple[int, int, int, int]
    dominant_image_idx: int

class SatelliteSegmenter:
    """
    A wrapper for the Mask2Former architecture fine-tuned for satellite imagery.
    Replaces internal 'FSU' and 'SigLIP' segmentation pipelines.
    """
    
    def __init__(self, model_id: str = "mfaytin/mask2former-satellite"):
        """Initializes the model and moves it to the appropriate device."""
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor: AutoImageProcessor = AutoImageProcessor.from_pretrained(model_id)
        self.model: Mask2FormerForUniversalSegmentation = (
            Mask2FormerForUniversalSegmentation.from_pretrained(model_id).to(self.device)
        )
        self.model.eval()

    @torch.no_grad()
    def segment(self, image: Image.Image) -> np.ndarray:
        """
        Performs semantic segmentation on a single image.
        
        Args:
            image: A PIL Image object.
            
        Returns:
            A 2D NumPy array (int64) where each pixel is a class ID.
        """
        inputs: Dict = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        
        # Post-process to map back to original image size
        prediction: torch.Tensor = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        
        return prediction.cpu().numpy()

class CLIPScreeningFilter:
    """
    Semantic screening module using CLIP. 
    Verifies if a detected segmentation change is visually confirmed by a VLM.
    """

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        """Loads CLIP processor and model."""
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: CLIPModel = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()

    @torch.no_grad()
    def is_valid_change(self, img1: Image.Image, img2: Image.Image, change: ChangeInstance) -> bool:
        """
        Validates the segmentation mask against the raw pixels using CLIP.
        
        Args:
            img1: Image at timestamp T1.
            img2: Image at timestamp T2.
            change: The ChangeInstance metadata containing the bounding box.
            
        Returns:
            Boolean: True if CLIP confirms the presence of the class in the region.
        """
        # Select the image where the object is predicted to be present
        target_img: Image.Image = img1 if change.dominant_image_idx == 1 else img2
        crop: Image.Image = target_img.crop(change.bbox)
        
        # Define prompts for zero-shot classification
        prompts: List[str] = [
            f"a high-resolution satellite view of {change.class_name}",
            "an empty plot of land or ground"
        ]
        
        inputs: Dict = self.processor(
            text=prompts, 
            images=crop, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        outputs = self.model(**inputs)
        probs: torch.Tensor = outputs.logits_per_image.softmax(dim=1)
        
        # Thresholding: If class probability is > 0.4, we consider it "verified"
        return float(probs[0][0].item()) > 0.4

class ChangeDetectionPipeline:
    """
    The main coordination class for the candidate generation and screening stages.
    """

    def __init__(self):
        self.segmenter: SatelliteSegmenter = SatelliteSegmenter()
        self.screener: CLIPScreeningFilter = CLIPScreeningFilter()

    def run_inference(self, img1: Image.Image, img2: Image.Image) -> List[ChangeInstance]:
        """
        The entry point for processing a pair of images.
        """
        # 1. Segmentation
        mask1: np.ndarray = self.segmenter.segment(img1)
        mask2: np.ndarray = self.segmenter.segment(img2)
        
        # 2. Pixel-level difference check
        diff_mask: np.ndarray = (mask1 != mask2)
        if diff_mask.sum() < MIN_TOTAL_CHANGED_PIXELS:
            return []

        # 3. Component Extraction & Semantic Screening
        final_candidates: List[ChangeInstance] = []
        for candidate in self._extract_instances(mask1, mask2, diff_mask):
            if self.screener.is_valid_change(img1, img2, candidate):
                final_candidates.append(candidate)
        
        return final_candidates

    def _extract_instances(self, m1: np.ndarray, m2: np.ndarray, diff: np.ndarray) -> Iterator[ChangeInstance]:
        """
        Finds connected components for each class and calculates change metrics.
        """
        for c_id, c_name in enumerate(SATELLITE_CLASSES):
            # Skip background/unreliable classes
            if c_name in ["ground_natural", "other_human_made"]:
                continue
            
            # Find union of the class across both images
            combined: np.ndarray = (m1 == c_id) | (m2 == c_id)
            labeled, num_features = ndimage.label(combined)
            
            for i in range(1, num_features + 1):
                region: np.ndarray = (labeled == i)
                if region.sum() < MIN_REGION_PIXELS:
                    continue
                
                # IoU Calculation
                m1_reg: np.ndarray = (m1 == c_id) & region
                m2_reg: np.ndarray = (m2 == c_id) & region
                
                union_px: int = int((m1_reg | m2_reg).sum())
                inter_px: int = int((m1_reg & m2_reg).sum())
                iou: float = inter_px / (union_px + 1e-8)
                
                changed_px: int = int((region & diff).sum())
                
                # Filter based on change significance
                if iou < IOU_THRESH and changed_px > MIN_CHANGED_PIXELS:
                    # Calculate Bounding Box
                    coords: np.ndarray = np.argwhere(region)
                    ymin, xmin = coords.min(axis=0)
                    ymax, xmax = coords.max(axis=0)
                    
                    yield ChangeInstance(
                        class_name=c_name,
                        class_id=c_id,
                        footprint=region,
                        iou=iou,
                        changed_px=changed_px,
                        bbox=(int(xmin), int(ymin), int(xmax), int(ymax)),
                        dominant_image_idx=1 if m1_reg.sum() > m2_reg.sum() else 2
                    )

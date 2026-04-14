"""Gemma 3-4B reward model library for Best-of-N selection in change detection."""

import logging
import re
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# Configure logging to provide clear pipeline tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BestOfNVerifier:
    """Runs Gemma 3-4B verification on image patches for Best-of-N ranking."""

    def __init__(self, model_id: str = "google/gemma-3-4b-it"):
        """
        Initializes the BestOfNVerifier with the official Gemma 3-4B model.

        Args:
            model_id: The official Hugging Face model identifier.
        """
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        logger.info("Loading Gemma 3-4B model from %s...", model_id)
        # Load multimodal processor and model as documented in Hugging Face
        self.processor: AutoProcessor = AutoProcessor.from_pretrained(model_id)
        self.model: Gemma3ForConditionalGeneration = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=self.dtype,
        ).eval()
        logger.info("Gemma 3-4B model loaded successfully on %s.", self.device)

    def _get_extended_patch(
        self, 
        image: Image.Image, 
        xmin: int, 
        ymin: int, 
        w: int, 
        h: int
    ) -> Image.Image:
        """
        Extracts an extended patch from a full image.

        Args:
            image: The source PIL image.
            xmin: X-coordinate of the bounding box.
            ymin: Y-coordinate of the bounding box.
            w: Width of the bounding box.
            h: Height of the bounding box.

        Returns:
            A cropped PIL image with 50% contextual expansion.
        """
        img_width, img_height = image.size

        # Expand bounding box by 50% in each direction
        x_expansion: float = np.ceil(w / 2)
        y_expansion: float = np.ceil(h / 2)
        
        expanded_xmin: int = int(max(0, xmin - x_expansion))
        expanded_ymin: int = int(max(0, ymin - y_expansion))
        expanded_xmax: int = int(min(img_width, xmin + w + x_expansion))
        expanded_ymax: int = int(min(img_height, ymin + h + y_expansion))

        return image.crop((expanded_xmin, expanded_ymin, expanded_xmax, expanded_ymax))

    def _get_images_by_class(
        self,
        query_class: str,
        examples_df: pd.DataFrame,
        target_size: Tuple[int, int],
    ) -> List[Image.Image]:
        """
        Retrieves few-shot reference images for a specific semantic class.

        Args:
            query_class: The semantic class name to filter for.
            examples_df: DataFrame containing reference image bytes and metadata.
            target_size: The (width, height) to resize images for the VLM.

        Returns:
            A list of PIL Images resized to target_size.
        """
        images: List[Image.Image] = []
        matching_df: pd.DataFrame = examples_df[examples_df['Class'] == query_class].head(5)

        for _, row in matching_df.iterrows():
            # Assume examples are stored as PIL-compatible bytes or objects
            # In a public repo, we use standard PIL opening logic
            from io import BytesIO
            full_image: Image.Image = Image.open(BytesIO(row['image_bytes'])).convert("RGB")
            
            patch: Image.Image = self._get_extended_patch(
                full_image, 
                int(row['xmin']), 
                int(row['ymin']), 
                int(row['width']), 
                int(row['height'])
            )
            images.append(patch.resize(target_size))

        return images

    def run_verification(
        self,
        img1: Image.Image,
        img2: Image.Image,
        change_metadata: Dict[str, Any],
        image_num_to_verify: int,
        examples_df: pd.DataFrame,
    ) -> Optional[int]:
        """
        Runs Gemma 3-4B verification on an image patch to generate a reward score.

        Args:
            img1: PIL image from timestamp 1.
            img2: PIL image from timestamp 2.
            change_metadata: Dict containing 'class_name', 'xmin', 'ymin', 'width', 'height'.
            image_num_to_verify: 1 or 2, indicating which image contains the class.
            examples_df: DataFrame with few-shot examples and scores.

        Returns:
            The reward score as an integer (1-5), or None if parsing fails.
        """
        selected_class: str = change_metadata['class_name']
        logger.info("Running Gemma verification for class: '%s'...", selected_class)
        target_size: Tuple[int, int] = (224, 224)

        # 1. Retrieve RAG Few-Shot Examples
        all_matching_images: List[Image.Image] = self._get_images_by_class(
            query_class=selected_class,
            examples_df=examples_df,
            target_size=target_size
        )
        
        matching_scores: List[int] = examples_df[
            examples_df['Class'] == selected_class
        ]['Score'].head(len(all_matching_images)).tolist()

        # 2. Construct the Interleaved Chat Prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": (
                            "You are an expert in recognizing objects from satellite images. "
                            f"Your task is to score a query image patch from 1 to 5 to verify if '{selected_class}' appears.\n\n"
                            "## Scoring Guide\n"
                            "* **5:** Definitely visible. Features and shadows are clear.\n"
                            "* **4:** Very likely. Features are mostly clear.\n"
                            "* **3:** Ambiguous. Likely does not contain the object.\n"
                            "* **2:** Unlikely.\n"
                            "* **1:** Definitely NOT present.\n"
                        )
                    }
                ]
            }
        ]

        # Add reference images to the user message
        images_for_gemma: List[Image.Image] = []
        for i, (img, score) in enumerate(zip(all_matching_images, matching_scores)):
            images_for_gemma.append(img)
            messages[0]["content"].extend([
                {"type": "text", "text": f"Example {i+1} (Score = {score}):"},
                {"type": "image"}
            ])

        # 3. Add the actual Query Patch
        query_base_img: Image.Image = img1 if image_num_to_verify == 1 else img2
        query_patch: Image.Image = self._get_extended_patch(
            query_base_img,
            change_metadata['xmin'],
            change_metadata['ymin'],
            change_metadata['width'],
            change_metadata['height']
        ).resize(target_size)
        
        images_for_gemma.append(query_patch)
        messages[0]["content"].extend([
            {"type": "text", "text": "Query Image:"},
            {"type": "image"},
            {"type": "text", "text": "Numerical Score (1-5):"}
        ])

        # 4. Model Inference
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            output_ids: torch.Tensor = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
            response: str = self.processor.decode(
                output_ids[0][inputs.input_ids.shape[-1]:], 
                skip_special_tokens=True
            )

        # 5. Reward Extraction
        match = re.search(r'[1-5]', response)
        if match:
            gemma_score: int = int(match.group())
            logger.info("Gemma reward score: %d", gemma_score)
            return gemma_score
        
        logger.warning("Could not parse score from response: %s", response)
        return None

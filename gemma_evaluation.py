"""
Library for zero-shot evaluation of satellite change detection using Gemma 3-4B.

This module provides a standardized evaluation interface for the research 
pipeline. It leverages the Hugging Face 'image-text-to-text' pipeline to 
perform temporal reasoning across image pairs and classify changes based on 
semantic significance.
"""

import io
import logging
import argparse
import torch
import pandas as pd
from PIL import Image
from typing import List, Dict, Any, Optional, Union
from transformers import pipeline

# Configure logging for experiment traceability
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GemmaEvaluator:
    """
    Evaluates semantic changes between image pairs using the Gemma 3-4B-it model.
    Utilizes the Transformers pipeline for multi-modal inference.
    """

    def __init__(self, model_id: str = "google/gemma-3-4b-it"):
        """
        Initializes the evaluation pipeline with Gemma 3.

        Args:
            model_id: The Hugging Face repository ID.
        """
        logger.info(f"Initializing Gemma evaluation pipeline for {model_id}...")
        
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Standard Hugging Face pipeline for multimodal tasks
        self.pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            device=self.device,
            torch_dtype=self.dtype,
            trust_remote_code=False
        )
        
        # Binary constraint system prompt
        self.system_prompt: str = (
            "You are a precise satellite imagery analyst. Your task is to compare two "
            "temporal images and detect significant semantic changes (e.g., new infrastructure, "
            "building demolition, or land clearing). "
            "Constraint: Answer ONLY with 'Yes' or 'No'. Do not explain your reasoning."
        )

    def _prepare_image(self, img_source: Union[bytes, str]) -> Image.Image:
        """Loads raw bytes or hex-encoded strings into a PIL object."""
        try:
            if isinstance(img_source, str):
                # Handle hex-encoded strings from the construction stage
                img_bytes = bytes.fromhex(img_source)
            else:
                img_bytes = img_source
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            logger.error(f"Image loading failed: {e}")
            raise

    def evaluate_pair(self, img1_data: Any, img2_data: Any) -> str:
        """
        Performs a single zero-shot classification on an image pair.

        Args:
            img1_data: Data for image at timestamp 1 (Before).
            img2_data: Data for image at timestamp 2 (After).

        Returns:
            A string: 'Yes' or 'No'.
        """
        img1: Image.Image = self._prepare_image(img1_data)
        img2: Image.Image = self._prepare_image(img2_data)

        # Construct the official Gemma 3 multimodal message format
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image 1 (Before):"},
                    {"type": "image"},
                    {"type": "text", "text": "Image 2 (After):"},
                    {"type": "image"},
                    {"type": "text", "text": "Is there a significant semantic change between these images?"}
                ]
            }
        ]

        # Inference
        # We pass the images via the images parameter in the pipeline call
        output = self.pipe(
            text=messages, 
            images=[img1, img2], 
            max_new_tokens=10,
            generate_kwargs={"do_sample": False} # Deterministic for evaluation
        )

        # Parse the last content from the assistant's turn
        raw_answer: str = output[0]["generated_text"][-1]["content"].strip().lower()
        
        if "yes" in raw_answer:
            return "Yes"
        return "No"

    def run_benchmark(self, csv_path: str, output_path: str):
        """
        Iterates through the evaluation dataset and saves predictions.

        Args:
            csv_path: Input CSV containing image data (hex-encoded) and ground truth.
            output_path: Destination for the results CSV.
        """
        logger.info(f"Running benchmark on {csv_path}")
        df: pd.DataFrame = pd.read_csv(csv_path)
        
        predictions: List[str] = []

        for idx, row in df.iterrows():
            logger.info(f"Processing sample {idx + 1}/{len(df)}...")
            
            # assumes 'img1_hex' and 'img2_hex' columns from construction stage
            result = self.evaluate_pair(row['img1_hex'], row['img2_hex'])
            predictions.append(result)

        df['prediction'] = predictions
        df.to_csv(output_path, index=False)
        logger.info(f"Benchmark finished. Results saved to {output_path}")

# --- Execution ---

def main():
    parser = argparse.ArgumentParser(description="Gemma 3 Zero-Shot Evaluation")
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="Model identifier")
    parser.add_argument("--input", required=True, help="Input CSV/JSONL path")
    parser.add_argument("--output", required=True, help="Output results CSV path")
    args = parser.parse_args()

    evaluator = GemmaEvaluator(model_id=args.model)
    evaluator.run_benchmark(args.input, args.output)

if __name__ == "__main__":
    main()

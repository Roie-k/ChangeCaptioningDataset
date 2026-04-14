"""
Library for zero-shot evaluation of satellite change detection using Gemini.

This module implements the evaluation stage of the pipeline, where the model 
is tasked with making binary decisions (Yes/No) on temporal image pairs to 
validate the accuracy of the proposed data-generation method.
"""

import json
import logging
import io
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from PIL import Image
from google import genai
from google.genai import types

# Configure logging for experimental tracking
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GeminiEvaluator:
    """
    Evaluates semantic changes between image pairs using the Gemini 3 API.
    Enforces strict binary output for objective metric calculation.
    """

    def __init__(self, api_key: str, model_id: str = "gemini-3-flash-preview"):
        """
        Initializes the evaluation client.

        Args:
            api_key: Official Google GenAI API Key.
            model_id: The specific model version used for the evaluation benchmark.
        """
        self.client: genai.Client = genai.Client(api_key=api_key)
        self.model_id: str = model_id
        
        # Strict instruction prompt for zero-shot evaluation
        self.system_instruction: str = (
            "You are a precise satellite imagery analyst. Compare the two provided images "
            "(Timestamp 1 and Timestamp 2) of the same geographical location.\n\n"
            "Task: Determine if there is a significant semantic change (e.g., new building, "
            "road construction, land clearing).\n\n"
            "Constraint: You MUST answer with exactly one word: 'Yes' or 'No'. "
            "Do not provide explanations or additional text."
        )

    def _load_image(self, img_source: Any) -> Image.Image:
        """Loads an image from bytes or hex string into a PIL object."""
        if isinstance(img_source, str):
            # Handle hex-encoded strings from the refactored JSONL/CSV
            img_bytes = bytes.fromhex(img_source)
        else:
            img_bytes = img_source
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def evaluate_pair(self, img1_data: Any, img2_data: Any) -> Optional[str]:
        """
        Performs a single evaluation call to the Gemini API.

        Args:
            img1_data: Bytes or hex string of the first image.
            img2_data: Bytes or hex string of the second image.

        Returns:
            A string ('Yes' or 'No') or None if the API call fails.
        """
        try:
            img1: Image.Image = self._load_image(img1_data)
            img2: Image.Image = self._load_image(img2_data)

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[
                    "Timestamp 1 (Before):", img1,
                    "Timestamp 2 (After):", img2,
                    self.system_instruction
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,  # Deterministic output for evaluation
                    max_output_tokens=5,
                )
            )
            
            # Clean response to ensure it's strictly Yes or No
            result: str = response.text.strip().capitalize()
            if "Yes" in result:
                return "Yes"
            elif "No" in result:
                return "No"
            return result
            
        except Exception as e:
            logger.error(f"Evaluation API Error: {e}")
            return None

    def run_benchmark(self, input_path: str, output_path: str):
        """
        Iterates through an evaluation set and saves predictions.

        Args:
            input_path: Path to the dataset file (CSV or JSONL).
            output_path: Path to save the evaluation results.
        """
        logger.info(f"Starting benchmark evaluation on {input_path}")
        
        # Load dataset (supports standard CSV format from previous steps)
        df: pd.DataFrame = pd.read_csv(input_csv) if input_path.endswith('.csv') else pd.read_json(input_path, lines=True)
        
        results: List[Dict[str, Any]] = []

        for idx, row in df.iterrows():
            logger.info(f"Evaluating sample {idx}...")
            
            # Use hex-encoded images from the 'dataset_construction.py' stage
            prediction: Optional[str] = self.evaluate_pair(row['img1_hex'], row['img2_hex'])
            
            results.append({
                "sample_id": row.get('sample_id', idx),
                "ground_truth": row.get('ground_truth', 'Unknown'),
                "prediction": prediction
            })
            
            # Adaptive throttling to stay within API rate limits
            time.sleep(0.5)

        # Save results for metric calculation (Precision/Recall/F1)
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Benchmark complete. Results saved to {output_path}")

# --- CLI Implementation ---

def main():
    parser = argparse.ArgumentParser(description="Gemini-based Change Detection Evaluation")
    parser.add_argument("--api_key", required=True, help="Official Google GenAI API Key")
    parser.add_argument("--input_data", required=True, help="Path to evaluation dataset")
    parser.add_argument("--output_results", required=True, help="Path to save predictions")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model version")
    args = parser.parse_args()

    evaluator = GeminiEvaluator(api_key=args.api_key, model_id=args.model)
    evaluator.run_benchmark(args.input_data, args.output_results)

if __name__ == "__main__":
    main()

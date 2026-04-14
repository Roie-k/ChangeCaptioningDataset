"""
Core library for transforming satellite change detection metadata into 
instruction-tuning pairs for Vision-Language Models.

This module handles the 'Dataset Construction' stage by:
1. Parsing raw change metadata from CSV/DataFrames.
2. Rendering visual prompts (bounding boxes) for localized changes.
3. Generating complex, spatially-aware natural language prompts for 
   Yes/No and Multiple Choice tasks.
"""

import json
import logging
import uuid
import io
import re
import ast
import argparse
from typing import List, Dict, Any, Optional, Tuple, Iterator, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# Configure logging for pipeline transparency
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Prompt Templates ---

NO_CHANGE_YES_NO_PROMPT: str = (
    "You are an expert in satellite imagery analysis and change detection. "
    "Examine these two images. There is **no significant difference** between them. "
    "Your task is to generate a single 'Yes/No' question that checks for changes. "
    "Since there are no changes, the answer must be 'No'.\n\n"
    "**Guidelines:**\n"
    "- **Strict Prohibition:** Do NOT refer to specific objects or land use types "
    "(e.g., never mention roads, buildings, vegetation).\n"
    "- **Focus:** The question must address the **overall** state of the pair.\n"
    "- **Vocabulary:** Use high-level terms like 'observable discrepancy,' 'visual variance,' "
    "or 'scene composition shift.'\n\n"
    "**Output Format:**\n**Question:** <Your holistic question>\n**Answer:** No"
)

CHANGE_YES_NO_PROMPT_TEMPLATE: str = (
    "You are an expert in fine-grained satellite imagery analysis. Focus strictly "
    "on the change involving the {cls}.\n\n"
    "**Context Instruction:** The red bounding box in the image is **strictly for "
    "your internal reference**. You must ignore its visual presence when generating "
    "the question.\n\n"
    "**Task:** Generate a detailed 'Yes/No' question describing the change to this "
    "object and its relative location.\n\n"
    "**Guidelines:**\n"
    "- **Spatial Description:** Do NOT mention the 'red box'. Translate location "
    "into natural language (e.g., 'in the northwest corner', 'near the main road').\n"
    "- **Detail:** Describe the specific state of the {cls}.\n"
    "- **Confidence:** Answer 'Yes' if the change is clear, otherwise 'I am not sure'.\n\n"
    "**Output Format:**\n**Question:** <Your question>\n**Answer:** <Yes/I am not sure>"
)

# --- Core Logic ---

class VisualGuideRenderer:
    """Handles the rendering of visual prompts (bounding boxes) on satellite patches."""

    @staticmethod
    def draw_bbox(
        img_bytes: bytes, 
        xmin: int, 
        ymin: int, 
        width: int, 
        height: int,
        color: str = 'red',
        line_width: int = 3
    ) -> Optional[bytes]:
        """
        Draws a red bounding box on the image to help the model localize the change.
        """
        try:
            img: Image.Image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            draw: ImageDraw.Draw = ImageDraw.Draw(img)
            
            xmax: int = xmin + width
            ymax: int = ymin + height
            
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=line_width)
            
            buffer: io.BytesIO = io.BytesIO()
            img.save(buffer, format='JPEG')
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Failed to render bounding box: {e}")
            return None

class DatasetConstructor:
    """
    Main pipeline to convert raw change detection metadata into 
    multimodal instruction-tuning samples.
    """

    def __init__(self, output_file: str):
        """
        Initializes the constructor.
        
        Args:
            output_file: Path to the .jsonl file where the dataset will be saved.
        """
        self.output_file: str = output_file
        self.renderer: VisualGuideRenderer = VisualGuideRenderer()

    def _safe_eval_bytes(self, val: Union[str, bytes]) -> bytes:
        """Safely parses byte strings from CSV storage."""
        if isinstance(val, bytes):
            return val
        try:
            # Handle 'nan' issues common in CSV exports
            val_clean: str = re.sub(r'\bnan\b', 'None', val)
            result: Any = ast.literal_eval(val_clean)
            return result if isinstance(result, bytes) else bytes(val, 'utf-8')
        except Exception:
            return bytes(str(val), 'utf-8')

    def _parse_list_field(self, val: str) -> List[Any]:
        """Parses string representations of lists (e.g., class names, coordinates)."""
        try:
            val_clean: str = re.sub(r'\bnan\b', 'None', val)
            result: Any = ast.literal_eval(val_clean)
            return result if isinstance(result, list) else [result]
        except Exception:
            return []

    def process_row(self, row: pd.Series) -> Iterator[Dict[str, Any]]:
        """
        Processes a single row of metadata and yields training examples.
        """
        # 1. Decode Images
        img1_bytes: bytes = self._safe_eval_bytes(row['img1_bytes'])
        img2_bytes: bytes = self._safe_eval_bytes(row['img2_bytes'])
        
        # 2. Parse Change Metadata
        class_names: List[str] = self._parse_list_field(row['change_class_name'])
        xmins: List[int] = self._parse_list_field(row['change_rect_xmin'])
        ymins: List[int] = self._parse_list_field(row['change_rect_ymin'])
        widths: List[int] = self._parse_list_field(row['change_rect_width'])
        heights: List[int] = self._parse_list_field(row['change_rect_height'])
        
        # Condition flag indicates if the change passed the 'Best-of-N' reward threshold
        condition_flags: List[bool] = self._parse_list_field(row['condition_flag'])
        
        indices_to_keep: List[int] = [i for i, flag in enumerate(condition_flags) if flag]

        if not indices_to_keep:
            # Generate 'No Change' examples
            yield self._format_sample(img1_bytes, img2_bytes, NO_CHANGE_YES_NO_PROMPT, "No", "yes_no")
        else:
            # Generate examples for each validated change
            for idx in indices_to_keep:
                if idx >= len(class_names): continue
                
                cls_name: str = class_names[idx]
                
                # Render the visual guide (the red box)
                img1_guided: Optional[bytes] = self.renderer.draw_bbox(
                    img1_bytes, xmins[idx], ymins[idx], widths[idx], heights[idx]
                )
                img2_guided: Optional[bytes] = self.renderer.draw_bbox(
                    img2_bytes, xmins[idx], ymins[idx], widths[idx], heights[idx]
                )
                
                if img1_guided and img2_guided:
                    prompt: str = CHANGE_YES_NO_PROMPT_TEMPLATE.format(cls=cls_name)
                    yield self._format_sample(img1_guided, img2_guided, prompt, "Yes", "yes_no")

    def _format_sample(
        self, 
        img1: bytes, 
        img2: bytes, 
        prompt: str, 
        answer: str, 
        task_type: str
    ) -> Dict[str, Any]:
        """Constructs the final dictionary for JSONL export."""
        return {
            "sample_id": str(uuid.uuid4()),
            "images": [img1.hex(), img2.hex()], # Hex encoding for JSON compatibility
            "prompt": prompt,
            "answer": answer,
            "task_type": task_type
        }

    def run(self, csv_path: str):
        """Runs the construction pipeline on a CSV file."""
        logger.info(f"Loading metadata from {csv_path}")
        df: pd.DataFrame = pd.read_csv(csv_path)
        
        count: int = 0
        with open(self.output_file, 'w') as f:
            for _, row in df.iterrows():
                for sample in self.process_row(row):
                    f.write(json.dumps(sample) + '\n')
                    count += 1
        
        logger.info(f"Successfully constructed {count} examples in {self.output_file}")

# --- CLI Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Anonymous Dataset Construction Pipeline")
    parser.add_argument("--input_csv", required=True, help="Path to validated changes CSV")
    parser.add_argument("--output_jsonl", required=True, help="Path to save VLM dataset")
    args = parser.parse_args()

    constructor = DatasetConstructor(args.output_jsonl)
    constructor.run(args.input_csv)

if __name__ == "__main__":
    main()

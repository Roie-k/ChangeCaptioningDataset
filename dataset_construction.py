import argparse
import ast
import io
import json
import logging
import re
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd
from PIL import Image, ImageDraw


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


QUESTION_YES_NO_PROMPT_TEMPLATE: str = (
    "You are an expert in fine-grained satellite imagery analysis. Focus "
    "strictly on the change involving the {cls}.\n\n"
    "**Context Instruction:** The red bounding box (or marked area) in the image "
    "is strictly for your internal reference to help you locate the "
    "specific change. You must ignore its visual presence when "
    "generating the question.\n\n"
    "**Task:** Generate a detailed 'Yes/No' question that specifically describes "
    "the change to this object and its location within the image.\n\n"
    "**Guidelines:**\n"
    "- **Spatial Description:** Do NOT mention the bounding box, red square, or "
    "marked area in your output. Instead, translate the location into "
    "natural language (e.g., \"in the top-left quadrant,\" \"adjacent to "
    "the main road,\" \"in the southern section,\" \"near the cluster of "
    "trees\").\n"
    "- **Detail:** The question must describe the specific "
    "action or state of the {cls} (e.g., \"Has the {cls} in the "
    "northwest corner been demolished?\", \"Is there a newly constructed "
    "{cls} near the center?\").\n"
    "- **Confidence Check:** Formulate the question assuming the visual evidence suggests "
    "\"Yes\". However, you must evaluate image clarity. If the change is blurry, "
    "occluded, or ambiguous, output \"I am not sure\" as the "
    "answer. If the change is clear, output \"Yes\".\n\n"
    "**Output Format:**\n"
    "**Question:** <Your detailed, spatially descriptive question>\n"
    "**Answer:** <\"Yes\" or \"I am not sure\">"
)

QUESTION_YES_NO_NO_CHANGE_PROMPT: str = (
    "You are an expert in satellite imagery analysis and change detection. "
    "Examine these two images. There is **no significant difference** "
    "between them. Your task is to generate a single 'Yes/No' question "
    "that checks for changes. Since there are no changes, the answer must "
    "be \"No\".\n\n"
    "**Guidelines:**\n"
    "- **Strict Prohibition:** Do NOT refer to specific objects or land use "
    "types (e.g., never mention roads, buildings, vegetation, vehicles, or "
    "construction).\n"
    "- **Focus:** The question must address the **overall** state of the "
    "pair.\n"
    "- **Vocabulary:** Vary your phrasing using high-level terms like "
    "\"observable discrepancy,\" \"visual variance,\" \"global alteration,\" "
    "\"detectable anomaly,\" \"scene composition shift,\" or \"temporal "
    "deviation.\"\n\n"
    "**Output Format:**\n"
    "Provide the output strictly in the following format, with no other text:\n"
    "**Question:** <Your holistic question here>\n"
    "**Answer:** No"
)

MULTIPLE_CHOICE_CHANGE_PROMPT_TEMPLATE: str = (
    "You are an expert in generating multiple-choice questions based on "
    "visual changes in satellite imagery.\n\n"
    "**Context Instruction:**\n"
    "The red bounding box in the image is strictly for your internal "
    "reference to help you locate the specific change involving the "
    "{cls}. You must ignore its visual presence when writing the "
    "question.\n\n"
    "**Guidelines:**\n"
    "- **No Leakage:** Do NOT mention the red box, bounding box, square, or marked area in the text. "
    "The user should not know a box exists.\n"
    "- **Spatial Description:** Instead, describe the location using natural language relative to "
    "the image frame or landmarks (e.g., \"in the northwest corner,\" "
    "\"adjacent to the main road,\" \"in the center\").\n"
    "- **The Question:** Ask specifically about the nature of the change to the "
    "{cls} at that described location.\n"
    "- **The Options:** Generate 4 options (A, B, C, D). One must accurately describe the change "
    "(e.g., \"It was demolished\"). The other 3 must be plausible distractors.\n"
    "- **Uncertainty Protocol:** If the image quality is poor, blurry, or the change is ambiguous, "
    "generate the question and options normally, but set the final answer to \"I am not sure\".\n\n"
    "**Output Format:**\n"
    "**Question:** <Your question with natural spatial context>\n"
    "**A)** <Option A>\n"
    "**B)** <Option B>\n"
    "**C)** <Option C>\n"
    "**D)** <Option D>\n"
    "**Answer:** <Letter> OR \"I am not sure\""
)

MULTIPLE_CHOICE_NO_CHANGE_PROMPT: str = (
    "You are an expert in satellite imagery analysis and change detection. "
    "Examine these two images. There is **no significant difference** "
    "between them. Your task is to generate a Multiple Choice question "
    "regarding the overall comparison of the images.\n\n"
    "**Guidelines:**\n"
    "- **Strict Prohibition:** Do NOT refer to specific objects, land use "
    "types, or distinctive features (e.g., avoid \"roads,\" \"buildings,\" "
    "\"vegetation,\" or \"infrastructure\").\n"
    "- **Question Focus:** Ask broadly about the scene's stability, "
    "consistency, or temporal coherence.\n"
    "- **Options:** Provide 4 options. Since there is no change, the "
    "**correct answer** must describe \"stability\" or \"lack of variance.\" "
    "The **distractors** (wrong answers) should describe generic levels of "
    "change (e.g., \"Significant global alteration,\" \"Moderate visual "
    "shift\") without describing *what* changed.\n"
    "- **Vocabulary:** Use abstract terms like \"structural integrity,\" "
    "\"visual parity,\" \"scene consistency,\" or \"temporal deviation.\"\n\n"
    "**Output Format:**\n"
    "Provide the output strictly in the following format:\n"
    "**Question:** <Your holistic question>\n"
    "**A)** <Option 1>\n"
    "**B)** <Option 2>\n"
    "**C)** <Option 3>\n"
    "**D)** <Option 4>\n"
    "**Answer:** <Correct Option Letter>"
)


class VisualGuideRenderer:
    """Renders a visual guide box for localized benchmark generation."""

    @staticmethod
    def draw_bbox(
        img_bytes: bytes,
        xmin: int,
        ymin: int,
        width: int,
        height: int,
        color: str = "red",
        line_width: int = 3,
    ) -> Optional[bytes]:
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            draw = ImageDraw.Draw(img)

            xmax = xmin + width
            ymax = ymin + height
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=line_width)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return buffer.getvalue()
        except Exception as exc:
            logger.error("Failed to draw bounding box: %s", exc)
            return None


class DatasetConstructor:
    """
    Construct benchmark-generation requests for:
    - yes_no
    - mcq

    Open-ended generation is intentionally not supported.
    """

    def __init__(self, output_file: str, keep_uncertain: bool = False) -> None:
        self.output_file = output_file
        self.keep_uncertain = keep_uncertain
        self.renderer = VisualGuideRenderer()

    @staticmethod
    def _safe_eval_bytes(val: Union[str, bytes]) -> bytes:
        if isinstance(val, bytes):
            return val
        try:
            val_clean = re.sub(r"\bnan\b", "None", str(val))
            parsed = ast.literal_eval(val_clean)
            if isinstance(parsed, bytes):
                return parsed
            if isinstance(parsed, bytearray):
                return bytes(parsed)
            return bytes(str(val), "utf-8")
        except Exception:
            return bytes(str(val), "utf-8")

    @staticmethod
    def _parse_list_field(val: Any) -> List[Any]:
        if isinstance(val, list):
            return val
        if pd.isna(val):
            return []
        try:
            val_clean = re.sub(r"\bnan\b", "None", str(val))
            parsed = ast.literal_eval(val_clean)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return []

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(round(float(value)))
        except Exception:
            return default

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        return re.sub(r"\s+", " ", answer.strip())

    def _format_sample(
        self,
        before_image: bytes,
        after_image: bytes,
        task_type: str,
        prompt: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "sample_id": str(uuid.uuid4()),
            "task_type": task_type,
            "before_image_hex": before_image.hex(),
            "after_image_hex": after_image.hex(),
            "prompt": prompt,
            "metadata": metadata,
        }

    def _build_yes_no_change_sample(
        self,
        img1_bytes: bytes,
        img2_bytes: bytes,
        cls_name: str,
        xmin: int,
        ymin: int,
        width: int,
        height: int,
        candidate_index: int,
    ) -> Optional[Dict[str, Any]]:
        img1_guided = self.renderer.draw_bbox(img1_bytes, xmin, ymin, width, height)
        img2_guided = self.renderer.draw_bbox(img2_bytes, xmin, ymin, width, height)

        if img1_guided is None or img2_guided is None:
            return None

        prompt = QUESTION_YES_NO_PROMPT_TEMPLATE.format(cls=cls_name)
        return self._format_sample(
            before_image=img1_guided,
            after_image=img2_guided,
            task_type="yes_no",
            prompt=prompt,
            metadata={
                "class_name": cls_name,
                "xmin": xmin,
                "ymin": ymin,
                "width": width,
                "height": height,
                "candidate_index": candidate_index,
                "is_no_change": False,
            },
        )

    def _build_mcq_change_sample(
        self,
        img1_bytes: bytes,
        img2_bytes: bytes,
        cls_name: str,
        xmin: int,
        ymin: int,
        width: int,
        height: int,
        candidate_index: int,
    ) -> Optional[Dict[str, Any]]:
        img1_guided = self.renderer.draw_bbox(img1_bytes, xmin, ymin, width, height)
        img2_guided = self.renderer.draw_bbox(img2_bytes, xmin, ymin, width, height)

        if img1_guided is None or img2_guided is None:
            return None

        prompt = MULTIPLE_CHOICE_CHANGE_PROMPT_TEMPLATE.format(cls=cls_name)
        return self._format_sample(
            before_image=img1_guided,
            after_image=img2_guided,
            task_type="mcq",
            prompt=prompt,
            metadata={
                "class_name": cls_name,
                "xmin": xmin,
                "ymin": ymin,
                "width": width,
                "height": height,
                "candidate_index": candidate_index,
                "is_no_change": False,
            },
        )

    def _build_yes_no_no_change_sample(
        self,
        img1_bytes: bytes,
        img2_bytes: bytes,
    ) -> Dict[str, Any]:
        return self._format_sample(
            before_image=img1_bytes,
            after_image=img2_bytes,
            task_type="yes_no",
            prompt=QUESTION_YES_NO_NO_CHANGE_PROMPT,
            metadata={
                "class_name": "no_change",
                "is_no_change": True,
            },
        )

    def _build_mcq_no_change_sample(
        self,
        img1_bytes: bytes,
        img2_bytes: bytes,
    ) -> Dict[str, Any]:
        return self._format_sample(
            before_image=img1_bytes,
            after_image=img2_bytes,
            task_type="mcq",
            prompt=MULTIPLE_CHOICE_NO_CHANGE_PROMPT,
            metadata={
                "class_name": "no_change",
                "is_no_change": True,
            },
        )

    def process_row(self, row: pd.Series) -> Iterator[Dict[str, Any]]:
        img1_bytes = self._safe_eval_bytes(row["img1_bytes"])
        img2_bytes = self._safe_eval_bytes(row["img2_bytes"])

        class_names = self._parse_list_field(row.get("change_class_name", []))
        xmins = self._parse_list_field(row.get("change_rect_xmin", []))
        ymins = self._parse_list_field(row.get("change_rect_ymin", []))
        widths = self._parse_list_field(row.get("change_rect_width", []))
        heights = self._parse_list_field(row.get("change_rect_height", []))
        condition_flags = self._parse_list_field(row.get("condition_flag", []))

        indices_to_keep = [i for i, flag in enumerate(condition_flags) if bool(flag)]

        if not indices_to_keep:
            yield self._build_yes_no_no_change_sample(img1_bytes, img2_bytes)
            yield self._build_mcq_no_change_sample(img1_bytes, img2_bytes)
            return

        for idx in indices_to_keep:
            if idx >= len(class_names):
                continue
            if idx >= len(xmins) or idx >= len(ymins) or idx >= len(widths) or idx >= len(heights):
                continue

            cls_name = str(class_names[idx])
            xmin = self._safe_int(xmins[idx])
            ymin = self._safe_int(ymins[idx])
            width = self._safe_int(widths[idx])
            height = self._safe_int(heights[idx])

            yes_no_sample = self._build_yes_no_change_sample(
                img1_bytes=img1_bytes,
                img2_bytes=img2_bytes,
                cls_name=cls_name,
                xmin=xmin,
                ymin=ymin,
                width=width,
                height=height,
                candidate_index=idx,
            )
            if yes_no_sample is not None:
                yield yes_no_sample

            mcq_sample = self._build_mcq_change_sample(
                img1_bytes=img1_bytes,
                img2_bytes=img2_bytes,
                cls_name=cls_name,
                xmin=xmin,
                ymin=ymin,
                width=width,
                height=height,
                candidate_index=idx,
            )
            if mcq_sample is not None:
                yield mcq_sample

    def run(self, csv_path: str) -> None:
        logger.info("Loading metadata from %s", csv_path)
        df = pd.read_csv(csv_path)

        count = 0
        with open(self.output_file, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                for sample in self.process_row(row):
                    f.write(json.dumps(sample) + "\n")
                    count += 1

        logger.info("Successfully wrote %d generation requests to %s", count, self.output_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="RSRCC benchmark construction prompt builder")
    parser.add_argument("--input_csv", required=True, help="Path to validated changes CSV")
    parser.add_argument("--output_jsonl", required=True, help="Path to save generation requests")
    args = parser.parse_args()

    constructor = DatasetConstructor(output_file=args.output_jsonl)
    constructor.run(args.input_csv)


if __name__ == "__main__":
    main()

import argparse
import pandas as pd
from pathlib import Path
from PIL import Image

from change_detection_core import ChangeDetectionCore
from best_of_n_retreival import BestOfNVerifier
from dataset_construction import DatasetConstructor


def load_image_pairs(input_dir: str):
    root = Path(input_dir)
    for before_path in sorted(root.glob("*_before.*")):
        pair_id = before_path.stem.rsplit("_before", 1)[0]
        after_matches = list(root.glob(f"{pair_id}_after.*"))
        if not after_matches:
            continue

        after_path = after_matches[0]
        img1 = Image.open(before_path).convert("RGB")
        img2 = Image.open(after_path).convert("RGB")
        yield pair_id, img1, img2


def image_to_bytes(img: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def run_pipeline(args):
    detector = ChangeDetectionCore(
        backend=args.segmentation_backend,
        model_id=args.segmentation_model if args.segmentation_model else None,
    )
    verifier = BestOfNVerifier(model_id=args.reward_model)
    constructor = DatasetConstructor(output_file=args.output_file)

    examples_df = pd.read_csv(args.examples_csv)
    rows = []

    for pair_id, img1, img2 in load_image_pairs(args.input_dir):
        candidates = detector.run_inference(img1, img2)

        validated = []
        for cand in candidates:
            score = verifier.run_verification(
                img1=img1,
                img2=img2,
                change_metadata={
                    "class_name": cand.class_name,
                    "xmin": cand.xmin,
                    "ymin": cand.ymin,
                    "width": cand.width,
                    "height": cand.height,
                },
                image_num_to_verify=cand.dominant_image_idx,
                examples_df=examples_df,
            )
            if score is not None and score >= args.reward_threshold:
                validated.append(cand)

        img1_bytes = image_to_bytes(img1)
        img2_bytes = image_to_bytes(img2)

        if not validated:
            rows.append(
                {
                    "pair_id": pair_id,
                    "img1_bytes": repr(img1_bytes),
                    "img2_bytes": repr(img2_bytes),
                    "change_class_name": str([]),
                    "change_rect_xmin": str([]),
                    "change_rect_ymin": str([]),
                    "change_rect_width": str([]),
                    "change_rect_height": str([]),
                    "condition_flag": str([]),
                }
            )
            continue

        for cand in validated:
            rows.append(
                {
                    "pair_id": pair_id,
                    "img1_bytes": repr(img1_bytes),
                    "img2_bytes": repr(img2_bytes),
                    "change_class_name": str([cand.class_name]),
                    "change_rect_xmin": str([cand.xmin]),
                    "change_rect_ymin": str([cand.ymin]),
                    "change_rect_width": str([cand.width]),
                    "change_rect_height": str([cand.height]),
                    "condition_flag": str([True]),
                }
            )

    temp_csv = Path(args.output_file).with_suffix(".tmp.csv")
    pd.DataFrame(rows).to_csv(temp_csv, index=False)
    constructor.run(str(temp_csv))

    if not args.keep_temp_csv:
        temp_csv.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSRCC pipeline")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--examples_csv", required=True)
    parser.add_argument("--output_file", default="prompt_requests.jsonl")
    parser.add_argument("--segmentation_backend", default="mask2former", choices=["mask2former", "segformer"])
    parser.add_argument("--segmentation_model", default=None)
    parser.add_argument("--reward_model", default="google/gemma-3-4b-it")
    parser.add_argument("--reward_threshold", type=int, default=4)
    parser.add_argument("--keep_temp_csv", action="store_true")
    args = parser.parse_args()

    run_pipeline(args)

# RSRCC: A Remote Sensing Regional Change Comprehension Benchmark Constructed via Retrieval-Augmented Best-of-N Ranking

![Data Examples](data_example.png)

This repository provides the official code and benchmark resources for **RSRCC**, a remote sensing benchmark for **localized semantic change reasoning** over bi-temporal satellite image pairs. Unlike prior datasets that mainly describe global before/after differences at the scene level, RSRCC is built around **localized, change-specific question answering**, where each example targets a particular semantic change instance.

The benchmark is constructed through a hierarchical semi-supervised curation pipeline that combines:
- semantic segmentation for candidate localization,
- image-text semantic screening with a remote-sensing-tuned vision-language encoder,
- retrieval-augmented **Best-of-N** validation for ambiguous cases,
- and LLM-based linguistic generation of region-grounded question-answer pairs.

**Dataset:** https://huggingface.co/datasets/google/RSRCC\

## Overview

Traditional change detection identifies **where** changes occur, typically through binary or semantic masks. RSRCC is designed to evaluate **what changed** at the level of a specific localized semantic instance. Each example asks the model to reason about a particular region or object rather than summarize the whole scene. This makes RSRCC a benchmark for:
- localized semantic change recognition,
- temporal comparison,
- region-grounded multimodal reasoning,
- and semantic disambiguation between meaningful changes and distractors. 

The current release contains **126k** examples, split into train, validation, and test sets. The dataset is intended as a benchmark resource for multimodal models operating on remote sensing imagery. 

---

## What RSRCC Measures

RSRCC is designed to measure:
- whether a specific localized semantic change is present,
- what semantic category best explains the observed change,
- whether the model can distinguish true changes from no-change distractors,
- and whether the model can answer grounded questions about a specific region across time. 

RSRCC is **not** designed to measure:
- open-vocabulary change discovery beyond the predefined class space,
- unrestricted global scene summarization,
- or reasoning over arbitrary unseen semantic categories. The current benchmark remains tied to a predefined semantic taxonomy and may underperform on novel or highly compositional changes outside this label space. 

---

## 📂 Dataset Structure

The dataset is organized into three standard splits:

- `train/`
- `val/`
- `test/`

Each split contains:

- `images/`  
  Bucketed folders containing deduplicated image files.
- `metadata.csv`  
  Metadata file linking image pairs to their textual annotation.

Each sample includes:

- a **before** image
- an **after** image
- a natural-language question and answer describing the semantic change

---

## 🖼️ Annotation Format

RSRCC contains three question formats:

- **Yes/No**  
  Binary questions verifying the presence or absence of a specific semantic change.

- **Multiple Choice (MCQ)**  
  Four-option questions asking which interpretation best explains the localized change.

Example question types include:
- “Has a new structure been built near the intersection?”
- “What change occurred to the building in the northeast part of the image?” 

Importantly, the LLM is **not** used to determine whether a change occurred. The semantic label is derived from the visual curation pipeline, while the LLM is used only to generate natural question-answer phrasings conditioned on the validated localized region.


---

## 🎯 Intended Use

RSRCC is intended for research on:

- semantic change captioning
- vision-language reasoning over remote sensing imagery
- multimodal question answering
- temporal scene understanding
- instruction tuning for remote sensing foundation models

---

## ⚠️ Notes

- The images are stored in bucketed subfolders for efficient hosting and repository limits.
- Image files are deduplicated so that repeated questions over the same scene pair reuse the same underlying before/after images.
- Metadata paths are relative to each split directory.

---

## 📥 Loading the Dataset

RSRCC can be loaded directly from the Hugging Face Hub using the `datasets` library.

```python
from datasets import load_dataset

dataset = load_dataset("google/RSRCC", split="train", streaming=True)

sample = next(iter(dataset))
print(sample.keys())
print(sample["text"])
```

Each sample provides three fields:

- before: the pre-change image
- after: the post-change image
- text: the semantic question-answer annotation

For quick inspection, we recommend using streaming=True, which allows reading a small number of samples without downloading the full split.


### 🖼️ Plot a Simple Example

The example below loads one sample and visualizes the temporal image pair.

```python
from datasets import load_dataset
import matplotlib.pyplot as plt

dataset = load_dataset("google/RSRCC", split="train", streaming=True)
sample = next(iter(dataset))

print(sample["text"])

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(sample["before"])
plt.title("Before")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sample["after"])
plt.title("After")
plt.axis("off")

plt.show()
```

---

---

## 🛰️ Overview

![Pipeline Overview](change_detection_pipeline_new_2-1.png)

Traditional change detection identifies *where* pixels changed; our pipeline explains *what* changed using natural language. The framework utilizes a novel hierarchical filtering approach to mitigate the high cost and inconsistency of manual annotation in the remote sensing domain.

### Key Contributions
* **Semi-supervised Pipeline:** An end-to-end framework that jointly supports change region identification and semantic description generation without requiring paired mask supervision.
* **Hierarchical Validation:** A two-step process combining fast semantic screening with reinforced Best-of-N ranking.
* **Reinforced Best-of-N Selection:** An optimization strategy where a large language model acts as a judge, guided by retrieval-augmented examples (RAG) of human annotations.

---

## 📂 Repository Structure

| File | Category | Description |
| :--- | :--- | :--- |
| `segmentation_core.py` | **Generation** | Localization of candidate changes ($\Delta x$) using connected component analysis on segmentation masks. |
| `semantic_screening.py` | **Screening** | Preliminary validation using a SigLIP Image-Text Encoder fine-tuned for remote sensing. |
| `best_of_n_retrieval.py`| **Ranking** | **Core Contribution:** Reinforced Best-of-N ranking formulation to resolve ambiguous cases. |
| `dataset_construction.py`| **Construction** | Automatic generation of instruction-tuning pairs (MCQ, Yes/No, Open-ended). |
| `main.py` | **Orchestration** | End-to-end execution script from raw multi-temporal scenes to the final dataset. |
| `gemini_evaluation.py` | **Evaluation** | API-based benchmark using Gemini 2.5 Flash. |
| `gemma_evaluation.py` | **Evaluation** | Local benchmark using Gemma-3-4B. |

---

## 🛠️ Installation

### Environment Setup
Ensure you are using Python 3.10+ and a GPU suitable for bfloat16 inference.

```bash
git clone https://github.com/Roie-k/ChangeCaptioningDataset.git
cd ChangeCaptioningDataset
pip install -r requirements.txt
```

### Authentication
1.  **Hugging Face:** Login to access public weights (Gemma 3):
    ```bash
    huggingface-cli login
    ```
2.  **Gemini API:** Export your key for automated captioning:
    ```bash
    export GOOGLE_API_KEY='your_api_key_here'
    ```

---

## 📖 Methodology: Reinforced Best-of-N

The core of our approach resolves semantic ambiguity by selecting an optimal change candidate through a retrieval-augmented preference model. 

1.  **Semantic Screening:** Candidates are discarded if the expected class does not appear in the Top-K predictions of the "After" image patch.
2.  **Best-of-N Ranking:** Ambiguous cases are forwarded to an LLM judge ($J_{\phi}$) that scores candidates ($1–5$) against retrieved RAG examples from human annotations.
3.  **Policy Improvement:** Only candidates that maximize the preference score or exceed a predefined threshold ($\tau$) are retained.

---

## 🖼️ Data Examples

The generated dataset includes visual reasoning questions for temporal satellite image pairs, capturing semantic changes such as new construction, demolition, or vegetation loss.

### Sample Output Format
Each sample consists of a temporal pair accompanied by a generated question:
* **Yes/No:** "Is there a new house visible now at the bottom-left of the cul-de-sac?".
* **Multiple-Choice:** captures discrete interpretations such as "Several new residential buildings have been constructed".

---

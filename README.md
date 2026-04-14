# Ranking the Changes: Reinforced Best-of-N Ranking for Semantic Change Captioning

![Data Examples](data_exmaple.pdf)

This repository provides the official implementation of the semi-supervised pipeline introduced in **"Ranking the Changes: Reinforced Best-of-N Ranking with Retrieval-Augmented Vision-Language Models for Semantic Change Captioning"**. This framework generates high-quality semantic change captioning datasets by aligning visual multi-temporal evidence with natural language descriptions.

---

## 🛰️ Overview

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
git clone https://github.com/anonymous-submission/ranking-the-changes.git
cd ranking-the-changes
pip install -r requirements.txt
```

### Authentication
1.  **Hugging Face:** Login to access public weights (Gemma 3, SigLIP 2):
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

## 🏃 Usage

### 1. Execute Pipeline
Run the full discovery pipeline on a subset of LEVIR-CD:
```bash
python main.py \
    --input_dir ./data/levir_sample \
    --examples_csv ./data/rag_human_annotations.csv \
    --output_file ./output/SCC_dataset.jsonl
```

### 2. Run Benchmarks
Evaluate model performance using established metrics (BLEU, BERTScore, CIDEr, SPICE):
```bash
python gemma_evaluation.py --input ./output/SCC_dataset.jsonl --output ./results/metrics.csv
```

---

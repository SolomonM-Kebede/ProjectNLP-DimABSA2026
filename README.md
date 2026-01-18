# Dimensional Aspect-Based Sentiment Analysis with Valence–Arousal  
### SemEval Subtasks 1, 2, and 3

This repository contains implementations and experiments for **Subtasks 1, 2, and 3** of an Aspect-Based Sentiment Analysis (ABSA) task with **valence–arousal (VA) prediction**.  
The work explores **model comparison**, **instruction-based generation**, and **pipeline-style optimization** for structured sentiment extraction.

---

## Overview

| Subtask | Description | Approach |
|-------|------------|---------|
| **Subtask 1** | Aspect–Opinion–VA extraction | Multiple encoder-based models |
| **Subtask 2** | Triplet extraction (Aspect, Opinion, VA) | FLAN-T5 (base / large) |
| **Subtask 3** | Quadruplet extraction (Aspect, Category, Opinion, VA) | FLAN-T5 (base / large) |

---

## Subtask 1: Model Comparison

**Goal:**  
Evaluate different pretrained language models for extracting aspect–opinion pairs with valence–arousal scores.

**Models tested:**
- BERT-based models
- RoBERTa-based models
- DeBERTa-based models
- Other transformer encoders

**Focus:**
- Model architecture comparison
- Extraction accuracy
- VA regression performance

Evaluation is conducted using **Precision, Recall, F1-score**, and **VA regression metrics**.

---

## Subtask 2: Triplet Extraction (Aspect, Opinion, VA)

**Models:**
- `flan-t5-base`
- `flan-t5-large`

**Approach:**
- Instruction-based text-to-text generation
- End-to-end structured JSON prediction
- Explicit handling of `NULL` values for missing elements

**Key aspects:**
- Stable structured decoding
- VA prediction robustness
- Optimization comparison between base and large models

---

## Subtask 3: Quadruplet Extraction (Aspect, Category, Opinion, VA)

**Models:**
- `flan-t5-base`
- `flan-t5-large`

**Approach:**
- Unified generative formulation of quadruplet extraction
- Category normalization using `ENTITY#ATTRIBUTE`
- NULL-aware decoding and evaluation
- Decoding and training optimization

**Evaluation metrics include:**
- Partial match (Aspect + Opinion)
- Full match (Aspect + Category + Opinion)
- RMSE for Valence and Arousal
- Concordance Correlation Coefficient (CCC)

---

## Pipeline Design

The system supports:
- **End-to-end generation**
- **Pipeline-style processing** (used for analysis and ablation)

`NULL` values are treated as **valid labels** during training and evaluation.

---

## Evaluation Metrics

- Precision / Recall / F1-score
- Partial vs Full match (Subtask 3)
- RMSE (Valence, Arousal)
- Concordance Correlation Coefficient (CCC)

---

## Repository Structure
```
├── data/                # JSONL datasets
├── datasets/            # Dataset classes
├── training/            # Training loops
├── evaluation/          # Metrics and evaluation scripts
├── models/              # Model configurations
├── utils/               # Helper functions
└── README.md
```

---

---

## Reproducibility

- Fixed random seeds
- Deterministic decoding settings where applicable
- Consistent evaluation protocol across all subtasks

---

## Dashboard & Visualization

A Streamlit-based interactive dashboard has been developed to visualize:

- Training and validation losses per epoch
- Step-level signal spectrum of training losses
- Best epoch highlighting
- Comparison of evaluation metrics (CCC, RMSE, F1, etc.) across models and datasets

### Run locally via Streamlit

```bash
# From project root
streamlit run stream-app.py
```
### Run via Docker 
#### Build the docker image
```bash
docker build -t solozk/dimaabsa-dashboard .
```
#### Run the container 
```bash
docker run -p 8501:8501 solozk/dimabsa-dashboard
```
Then open the browser http://localhost:8501

---

## Quick Links 
- [Streamlit Dashboard](https://solomonm-kebede-projectnlp-dimabsa2026-stream-app-cgvtf8.streamlit.app/)
- Docker Hub Image: ```bash solozk/dimabsa-dashboard```

---
## Notes

- Subtask 1 focuses on **model comparison**
- Subtasks 2 and 3 focus on **structured generation and optimization**
- This repository is designed for experimentation and analysis

---

## Citation

If you use this repository or build upon it, please cite the corresponding SemEval task and this work.

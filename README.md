# Aspect-Based Sentiment Analysis with Valence–Arousal  
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
- NULL prediction rate analysis

---

## Repository Structure

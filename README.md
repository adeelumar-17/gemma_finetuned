# Insurance LLM Demo 🛡️

A Streamlit app to run inference on and compare three insurance-domain language models,
including two fine-tuned by me on Gemma 4 E4B.

## Live Demo

```bash
streamlit run app.py
```

---

## Models

| Model | Base | Dataset | Type |
|---|---|---|---|
| Gemma-4-E4B · Bitext | google/gemma-4-E4B-it | [Bitext Insurance Chatbot](https://huggingface.co/datasets/bitext/Bitext-insurance-llm-chatbot-training-dataset) | Fine-tuned (QLoRA) |
| Gemma-4-E4B · InsuranceQA | google/gemma-4-E4B-it | [InsuranceQA v2](https://github.com/shuzi/insuranceQA/tree/master/V2) | Fine-tuned (QLoRA) |
| Llama-3-8B · Open-Insurance | Llama-3-8B | Mixed insurance corpus | Public baseline |

---

## Features

- **Single Model** tab — select any model, tweak inference settings, ask a question
- **Side-by-Side Compare** tab — send the same prompt to two models simultaneously with diff stats
- **Configurable inference** — temperature, top-p, top-k, repetition penalty, max tokens, 4-bit quantisation

---

## Setup

### Requirements
- Python 3.10–3.13
- GPU with ≥ 8 GB VRAM recommended (CPU works but is very slow)
- HuggingFace account (token needed if adapter repos are private)

### Install

```bash
pip install -r requirements.txt

# For faster Gemma-4 loading (Linux GPU only):
pip install unsloth unsloth_zoo
```

### Run

```bash
streamlit run app.py
```

---

## Training Details

Both Gemma models were fine-tuned using **QLoRA** on **Kaggle (T4 × 2)**:

| Hyperparameter | Bitext | InsuranceQA |
|---|---|---|
| LoRA rank | 16 | 16 |
| LoRA alpha | 32 | 32 |
| Learning rate | 1e-4 | 1e-4 |
| Max steps | 1950 | ~2412 (3 epochs) |
| Context length | 8192 | 8192 |
| Quantisation | 4-bit (QLoRA) | 4-bit (QLoRA) |

The task objective: **generate synthetic insurance data records** that match the
distribution, ranges, and characteristics of the training dataset.

---

## Notebooks

| File | Description |
|---|---|
| `bitext-gemma.ipynb` | Main training notebook for the Bitext model |
| `bitext-final.ipynb` | Continued evaluation + generation notebook for Bitext |
| `insurance-gemma.ipynb` | Main training notebook for the InsuranceQA model |
| `insurance-gemma-final.ipynb` | Continued training + evaluation notebook for InsuranceQA |

---

## File Structure

```
gemma_finetuned/
├── app.py                    ← Streamlit app (this is what you run)
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
├── bitext-gemma.ipynb        ← Bitext training notebook
├── bitext-final.ipynb        ← Bitext evaluation/generation notebook
├── insurance-gemma.ipynb     ← InsuranceQA training notebook
└── insurance-gemma-final.ipynb ← InsuranceQA evaluation/generation notebook
```


---
title: Iris Medical Ai
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: docker
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
---

# Iris Medical AI 

A voice-interactive medical diagnosis assistant that combines a fine-tuned LLM with an XGBoost classifier to deliver confidence-gated, explainable diagnoses from natural language symptom descriptions.

> **Try it live:** Coming soon

---

## Overview

Iris is built around a two-model pipeline: a classical ML classifier narrows the hypothesis space, and a fine-tuned LLM reasons over the top predictions to produce a final diagnosis and treatment recommendation. The two models act as a checks-and-balance system — low-confidence ML predictions are caught before reaching the LLM, and all outputs include a safety disclaimer.

This is v3 of the Iris project. [v2](https://github.com/Mystique03/iris-improved) used Groq's Llama 3.3 70B API for LLM inference. v3 replaces it with a domain-specific model fine-tuned on 10,000 medical consultations, self-hosted on HuggingFace Spaces.

---


## ML Model — XGBoost Classifier

| Property | Detail |
|---|---|
| Algorithm | XGBoost (`XGBClassifier`) |
| Classes | 16 diseases |
| Features | 132 binary symptom flags |
| Tuning | Optuna (50 trials, 8 hyperparameters) |
| Validation | 20% split from `Training.csv` |
| Evaluation | Held-out `Testing.csv` |


---

## LLM — Fine-tuned Qwen3-4B

| Property | Detail |
|---|---|
| Base model | `unsloth/Qwen3-4B-Instruct` |
| Method | QLoRA (4-bit) via Unsloth |
| Dataset | `lavita/ChatDoctor-HealthCareMagic-100k` (10k subset) |
| Training | 1 epoch, 1000 steps |
| Quantization | GGUF Q4_K_M |
| Deployment | HuggingFace Spaces |
| Tracking | Weights & Biases |

The model is trained only on assistant responses (`train_on_responses_only`) using the Qwen3 chat template

**Fine-tuning repo:** [Mystique03/medical-qwen3-4b-lora](https://huggingface.co/Mystique03/medical-qwen3-4b-lora)

---

## Guardrails

Two lightweight guardrails are applied without any external framework:

**Input — confidence gate:** If XGBoost's top prediction confidence is below 40%, Iris skips the LLM entirely and redirects the user to consult a doctor. Prevents the LLM from reasoning over low-signal input.

**Output — safety disclaimer:** Every LLM response is appended with a disclaimer reminding the user this is not a substitute for professional medical advice.

---

## Project Structure

    iris-medical-ai/
    ├── data/
    │   ├── Training.csv         
    │   ├── Testing.csv           
    │   └── best_params.json      
    ├── iris/
    │   ├── model.py              # XGBoost training with Optuna
    │   ├── llm.py                # HF Spaces inference + confidence-gated routing
    │   ├── voice.py              # STT via SpeechRecognition
    │   └── prompts.py            # All prompt templates
    ├── main.py                   
    ├── pyproject.toml
    ├── .env.example
    └── README.md

---

## Setup

**Requirements:** Python 3.11+, a microphone, [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/Mystique03/iris-medical-ai
cd iris-medical-ai
uv sync
cp .env.example .env
# Fill in HF_SPACE_URL in .env
uv run main.py
```

---

## Design Decisions

**Why fine-tune instead of prompt-engineering a larger model?**
A 70B model via API (Groq) gives strong general responses but has no domain adaptation. A fine-tuned 4B model on 10k medical consultations produces more structured, medically-grounded responses at zero API cost and no rate limits.

**Why keep XGBoost alongside the LLM?**
The LLM alone has no grounding mechanism — it can hallucinate diseases. XGBoost provides a structured, deterministic prior over 16 classes. The LLM's job is reasoning and explanation, not classification from scratch. This hybrid approach is more robust than either model alone.

---

## Limitations

- Symptom matching is substring-based — "chest pain" must appear verbatim in speech output
- 16 disease classes only — conditions outside this set will be misclassified
- Not a medical device — for educational purposes only

---

## Previous Version

[iris-improved (v2)](https://github.com/Mystique03/iris-improved) — single XGBoost prediction + Groq API (Llama 3.3 70B) for treatment recommendations.

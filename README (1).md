# PRODIGY_GA_01

## GPT-2 Text Generation Model (Fine-Tuned)

This project demonstrates fine-tuning the GPT-2 language model to generate coherent and contextually relevant text based on a custom dataset.

### Features
- Uses OpenAI GPT-2 via Hugging Face Transformers
- Fine-tuned on a custom text dataset
- Generates meaningful text from a given prompt

### Files
- `train_gpt2.py` – Script to fine-tune GPT-2
- `generate.py` – Script to generate text using trained model
- `train.txt` – Custom dataset
- `my_gpt2/` – Saved trained model

### How to Run
```bash
pip install transformers datasets torch accelerate
python train_gpt2.py
python generate.py

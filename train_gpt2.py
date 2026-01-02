from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

print("Loading tokenizer and model...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

print("Loading dataset...")
# STEP 1: Import libraries
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch

print("Loading tokenizer and model...")

# STEP 2: Load GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# GPT-2 needs pad token
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# STEP 3: Load dataset
print("Loading dataset...")
dataset = load_dataset("text", data_files={"train": "train.txt"})

# STEP 4: Tokenization function (THIS WAS MISSING)
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

# STEP 5: Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# IMPORTANT: labels = input_ids (THIS FIXES YOUR LOSS ERROR)
tokenized_dataset = tokenized_dataset.map(
    lambda x: {"labels": x["input_ids"]},
    batched=True
)

# STEP 6: Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_steps=1,
    save_steps=5,
    save_total_limit=1,
    report_to="none"
)

# STEP 7: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

# STEP 8: Train
print("Starting training...")
trainer.train()

# STEP 9: Save model
model.save_pretrained("my_gpt2")
tokenizer.save_pretrained("my_gpt2")

print("Training completed and model saved!")

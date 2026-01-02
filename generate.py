from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("Loading trained model...")

tokenizer = GPT2Tokenizer.from_pretrained("my_gpt2")
model = GPT2LMHeadModel.from_pretrained("my_gpt2")

prompt = "Artificial intelligence is"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=80,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated Text:")
print(generated_text)

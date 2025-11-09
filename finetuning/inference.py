import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------
# Config
# ----------------------
MODEL_DIR = "./flan-t5-finetuned"
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ----------------------
# Load model and tokenizer
# ----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)

# ----------------------
# Inference loop
# ----------------------
while True:
    question = input("You: ").strip()
    if question.lower() in ["exit", "quit"]:
        break

    # Add instruction prompt
    input_text = f"Answer the question: {question}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True
    ).to(device)

    output_ids = model.generate(
        **inputs,
        max_length=MAX_TARGET_LENGTH,
        num_beams=4,
        early_stopping=True
    )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Bot: {answer}\n")

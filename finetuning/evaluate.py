# evaluate.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_metric
from dataset_loader import load_dataset
from config import OUTPUT_DIR, MAX_LENGTH, DEVICE

# Load fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR).to(DEVICE)

dataset = load_dataset().train_test_split(test_size=0.1)["test"]
metric = load_metric("rouge")

for sample in dataset:
    inputs = tokenizer(sample["input_text"], return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    outputs = model.generate(**inputs, max_length=MAX_LENGTH)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    metric.add(prediction=pred, reference=sample["target_text"])

result = metric.compute()
print("📊 Evaluation metrics:", result)

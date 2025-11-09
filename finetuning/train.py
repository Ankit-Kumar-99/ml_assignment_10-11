import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset

# ----------------------
# Config
# ----------------------
MODEL_NAME = "google/flan-t5-small"  # Open model
DATA_PATH = "../dataset_creation/data/dataset_with_intents.csv"
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 3
OUTPUT_DIR = "./flan-t5-finetuned"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ----------------------
# Load tokenizer and model
# ----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

# ----------------------
# Load dataset
# ----------------------
dataset = load_dataset("csv", data_files=DATA_PATH)
train_dataset = dataset["train"] if "train" in dataset else dataset["train"]

# ----------------------
# Preprocessing function
# ----------------------
def preprocess(batch):
    # Add instruction prompt for FLAN-T5
    inputs = ["Answer the question: " + q for q in batch["question"]]
    targets = batch["answer"]

    model_inputs = tokenizer(
        inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length"
    )

    labels = tokenizer(
        targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length"
    ).input_ids

    # Replace pad token IDs with -100 for loss computation
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in seq] for seq in labels]
    model_inputs["labels"] = labels
    return model_inputs

tokenized_dataset = train_dataset.map(
    preprocess,
    batched=True,
    remove_columns=train_dataset.column_names
)

# ----------------------
# Training arguments
# ----------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    push_to_hub=False
)

# ----------------------
# Trainer
# ----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# ----------------------
# Train
# ----------------------
trainer.train()

# ----------------------
# Save model
# ----------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model fine-tuned and saved at {OUTPUT_DIR}")

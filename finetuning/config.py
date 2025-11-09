# config.py
DATASET_PATH = "../dataset_creation/data/dataset_with_intents.csv"  # your CSV from section 2
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # small LLaMA-2 model
OUTPUT_DIR = "./finetuned_model"
MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 5e-5
DEVICE = "cuda"  # "cpu" if GPU not available
LOGGING_STEPS = 50

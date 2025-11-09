# dataset_creation/config.py

from pathlib import Path

# --- General settings ---
DATA_DIR = Path("data")
MODEL_NAME = "gpt-5-nano"
TEMPERATURE = 0.0
LOGGING_INTERVAL = 20

# --- Paths ---
CLEANED_DATA_PATH = Path("../web_scrapping/data/cleaned_data.csv")
INTENT_DATA_PATH = DATA_DIR / "dataset_with_intents.csv"
STRUCTURED_DATA_PATH = DATA_DIR / "structured_dataset.jsonl"

# --- LLM + API Settings ---
INTENT_LABELS = [
    "greeting",
    "pricing",
    "setup",
    "troubleshooting",
    "feature_information",
    "general_query",
]
OPENAI_API_KEY=""
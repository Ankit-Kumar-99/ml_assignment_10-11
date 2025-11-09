"""
cleaner.py — Clean and preprocess scraped Stack Overflow Q&A data.

This script:
1. Loads raw scraped JSON data.
2. Cleans HTML entities and text noise.
3. Removes duplicates and malformed entries.
4. Categorizes unclean records (for debugging and transparency).
5. Saves cleaned and uncleaned datasets separately as CSV files.

Configuration is loaded from config.json for flexible setup.
"""

import json
import pandas as pd
import re
from pathlib import Path
from utils import load_config


# ==============================
# 🔧 Load Configuration
# ==============================
config = load_config()

RAW_PATH = Path(config["output_folder"]) / config["raw_file"]
CLEAN_PATH = Path(config["output_folder"]) / config["clean_file"]
UNCLEAN_PATH = Path(config["output_folder"]) / config["unclean_file"]
MIN_ANSWER_LEN = config.get("min_answer_length", 15)


# ==============================
# 🧹 Text Cleaning Utilities
# ==============================
def clean_text(text: str) -> str:
    """
    Remove unwanted artifacts such as HTML entities, extra spaces,
    and special symbols from the given text.

    Args:
        text (str): Input text string (raw scraped content).

    Returns:
        str: Cleaned and normalized text.
    """
    if not text:
        return ""
    text = re.sub(r"&\w+;", " ", text)  # Remove HTML entities (e.g., &nbsp;)
    text = re.sub(r"\s+", " ", text)    # Normalize whitespace
    return text.strip()


# ==============================
# ⚙️ Categorization Logic
# ==============================
def categorize_unclean(row: pd.Series) -> str:
    """
    Determine why a given Q&A entry should be marked as 'unclean'.

    Args:
        row (pd.Series): A row containing 'question', 'answer', and 'url'.

    Returns:
        str: Category of unclean reason (e.g., 'No Answer', 'Too Short').
    """
    ans = str(row.get("answer", "")).strip()
    q = str(row.get("question", "")).strip()
    url = row.get("url", "")

    # 1️⃣ Missing or null answer
    if ans.lower() in ["", "none", "nan"]:
        return "No Answer"

    # 2️⃣ Malformed or missing URL
    if not isinstance(url, str) or not url.startswith("https"):
        return "Malformed Entry"

    # 3️⃣ Answer too short (below threshold)
    if len(ans) < MIN_ANSWER_LEN:
        return "Too Short"

    # 4️⃣ Fallback — everything looks valid
    return "Other"


# ==============================
# 🧽 Main Cleaning Process
# ==============================
def clean_dataset():
    """Load, clean, filter, and save the Stack Overflow dataset."""
    # Step 1: Load raw JSON data
    if not RAW_PATH.exists():
        print("❌ raw_data.json not found. Please run scraper.py first.")
        return

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    total_raw = len(df)
    print(f"📥 Loaded {total_raw} raw entries from {RAW_PATH}")

    # Step 2: Clean text fields
    for col in ["question", "body", "answer"]:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(clean_text)

    # Step 3: Drop duplicates
    df.drop_duplicates(subset=["url"], inplace=True)

    # Step 4: Categorize each entry (clean vs unclean)
    df["reason"] = df.apply(categorize_unclean, axis=1)

    # Step 5: Filter clean entries
    clean_df = df[
        (df["reason"] == "Other")
        & (df["answer"].notna())
        & (df["answer"].str.strip() != "")
        & (df["answer"].str.len() >= MIN_ANSWER_LEN)
    ].copy()

    # Step 6: Collect uncleaned entries for debugging
    unclean_df = df[~df.index.isin(clean_df.index)].copy()

    # Step 7: Save clean and unclean datasets
    output_folder = Path(config["output_folder"])
    output_folder.mkdir(parents=True, exist_ok=True)

    clean_df.to_csv(CLEAN_PATH, index=False, encoding="utf-8")
    unclean_df.to_csv(UNCLEAN_PATH, index=False, encoding="utf-8")

    # Step 8: Display summary stats
    total_clean = len(clean_df)
    total_unclean = len(unclean_df)

    print("\n✅ Cleaning Complete!")
    print(f"   • Cleaned dataset saved to:   {CLEAN_PATH}")
    print(f"   • Uncleaned dataset saved to: {UNCLEAN_PATH}")
    print("\n📊 SUMMARY:")
    print(f"   • Total raw entries: {total_raw}")
    print(f"   • Cleaned entries:   {total_clean}")
    print(f"   • Uncleaned entries: {total_unclean}")

    # Step 9: Breakdown of reasons for uncleaned entries
    print("\n🧩 Breakdown of Uncleaned Entries:")
    print(unclean_df["reason"].value_counts().to_string())


# ==============================
# 🚀 Run as Script
# ==============================
if __name__ == "__main__":
    clean_dataset()

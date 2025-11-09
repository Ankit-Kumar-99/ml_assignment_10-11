"""
intent_classifier.py
--------------------
LLM-only intent classification for technical Q&A.

- Uses OpenAI LLM (gpt-5-mini or similar) to classify text into predefined intents.
- Strict hierarchical rules:
    1. Always pick the most relevant intent first.
    2. Only fallback to 'general_query' if nothing matches.
- Handles long question+body inputs, truncates for LLM if necessary.

Author: Your Name
"""

import os
import pandas as pd
from openai import OpenAI
from config import (
    OPENAI_API_KEY,
    MODEL_NAME,
    INTENT_LABELS,
    CLEANED_DATA_PATH,
    INTENT_DATA_PATH,
    LOGGING_INTERVAL,
)

# ------------------------------------------
# Initialize OpenAI client
# ------------------------------------------
if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OpenAI API key. Please check your config.py or .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)


# ------------------------------------------
# LLM-based intent classifier
# ------------------------------------------
def classify_intent_llm(question: str, body: str) -> str:
    """
    Classify a question + body into predefined intents using LLM.
    Strict hierarchy: always pick the most relevant intent first.
    """
    if not question or len(question.strip()) < 5:
        return "general_query"

    # Truncate long body to 500 chars for LLM
    body_summary = (body[:500] + "...") if body and len(body) > 500 else body or ""

    prompt = f"""
You are an expert technical Q&A assistant. You will classify the input into one of these intents:
{', '.join(INTENT_LABELS)}

Steps:
1. Carefully read the Question and Body.
2. Analyze the technical context (code snippets, frameworks, data structures, etc.).
3. Compare it with each intent and determine the closest match.
4. If it matches one intent clearly, return ONLY that intent.
5. Only if there is absolutely no match, fallback to "general_query".
6. Respond with a single intent label, lowercase, no punctuation, no explanation.

Input:
Question: "{question}"
Body: "{body_summary}"
"""

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            temperature=0.2
        )
        intent = response.output_text.strip().lower()
        return intent if intent in INTENT_LABELS else "general_query"

    except Exception as e:
        print(f"⚠️ LLM classification failed: {e}")
        return "general_query"


# ------------------------------------------
# Main processing pipeline
# ------------------------------------------
def add_intents_to_dataset():
    """
    Reads cleaned Q&A CSV, classifies each row into an intent,
    and saves the result as a new CSV for downstream processing.
    """
    print(f"📥 Loading dataset from: {CLEANED_DATA_PATH}")
    df = pd.read_csv(CLEANED_DATA_PATH)

    intents = []
    for i, row in df.iterrows():
        question = row.get("question", "")
        body = row.get("body", "")
        intent = classify_intent_llm(question, body)
        intents.append(intent)

        # Log progress
        if (i + 1) % LOGGING_INTERVAL == 0:
            print(f"🔹 Processed {i + 1}/{len(df)} entries...")

    # Save results
    df["intent"] = intents
    os.makedirs(os.path.dirname(INTENT_DATA_PATH), exist_ok=True)
    df.to_csv(INTENT_DATA_PATH, index=False, encoding="utf-8")

    print(f"\n✅ Intent classification complete. Saved to: {INTENT_DATA_PATH}")
    print("📊 Intent distribution:")
    print(df["intent"].value_counts())


# ------------------------------------------
# Run module
# ------------------------------------------
if __name__ == "__main__":
    add_intents_to_dataset()

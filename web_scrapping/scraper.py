"""
🧠 Stack Overflow Web Scraper
----------------------------------
Scrapes question–answer pairs from Stack Overflow
using BeautifulSoup and saves structured data as JSON.

Configuration such as base URLs, sleep durations, and headers
is loaded dynamically from `config.json`.
"""

import requests
import json
from pathlib import Path
from bs4 import BeautifulSoup
from utils import get_headers, rate_limit, load_config


# =========================================================
# 🔧 Load Configuration
# =========================================================
config = load_config()
BASE_URL = config["base_url"]
RESULTS = []


# =========================================================
# 🧩 Function: scrape_question_page
# =========================================================
def scrape_question_page(url):
    """
    Scrape a single Stack Overflow question page.

    Extracts:
        • Question title
        • Question body text
        • Accepted or top answer text (if any)

    Args:
        url (str): The URL of the question page to scrape.

    Returns:
        dict: A dictionary with URL, question, body, and answer.
    """
    # Send GET request for the question page
    res = requests.get(url, headers=config["headers"])
    soup = BeautifulSoup(res.text, "lxml")

    # --- Extract Question Title ---
    title_tag = soup.select_one("h1 a")
    question_title = title_tag.get_text(strip=True) if title_tag else "Untitled Question"

    # --- Extract Question Body ---
    body_tag = soup.select_one(".s-prose")
    question_body = body_tag.get_text(" ", strip=True) if body_tag else ""

    # --- Extract Accepted or Top Answer ---
    answer_tag = (
        soup.select_one(".answer.accepted-answer .s-prose")
        or soup.select_one(".answer .s-prose")
    )
    answer_text = answer_tag.get_text(" ", strip=True) if answer_tag else None

    # Return structured data
    return {
        "url": url,
        "question": question_title,
        "body": question_body,
        "answer": answer_text,
    }


# =========================================================
# 🧭 Function: scrape_stackoverflow
# =========================================================
def scrape_stackoverflow(pages=None):
    """
    Scrape multiple Stack Overflow pages for question–answer pairs.

    Handles pagination, rate limiting, and saves results to JSON.

    Args:
        pages (int, optional): Number of pages to scrape.
                               Defaults to config['scrape_pages'] if None.
    """
    pages = pages or config.get("scrape_pages", 5)
    print(f"🚀 Starting Stack Overflow scrape ({pages} pages)...\n")

    # Loop through paginated listing pages
    for page in range(1, pages + 1):
        print(f"🔎 Scraping list page {page}...")
        list_url = f"{BASE_URL}?tab=newest&page={page}"

        # --- Fetch List Page ---
        try:
            response = requests.get(list_url, headers=config["headers"])
            response.raise_for_status()
        except Exception as e:
            print(f"⚠️ Error fetching list page {page}: {e}")
            continue

        soup = BeautifulSoup(response.text, "lxml")

        # --- Extract Question Links ---
        # Updated CSS selector (verified for 2025)
        questions = soup.select("div.s-post-summary h3 a.s-link")
        print(f"   ↳ Found {len(questions)} questions")

        # --- Visit Each Question Link ---
        for q in questions:
            q_url = "https://stackoverflow.com" + q["href"]

            try:
                qa_pair = scrape_question_page(q_url)
                RESULTS.append(qa_pair)
            except Exception as e:
                print(f"⚠️ Error scraping {q_url}: {e}")

            # Respect rate limits between each question
            rate_limit(*config["sleep_between_questions"])

        # Respect rate limits between each list page
        rate_limit(*config["sleep_between_pages"])

    # =========================================================
    # 💾 Save Scraped Results
    # =========================================================
    output_folder = Path(config["output_folder"])
    output_folder.mkdir(parents=True, exist_ok=True)

    raw_path = output_folder / config["raw_file"]

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(RESULTS)} question-answer pairs to {raw_path}")


# =========================================================
# 🚀 Entry Point
# =========================================================
if __name__ == "__main__":
    scrape_stackoverflow()

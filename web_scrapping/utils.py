import random, time
from fake_useragent import UserAgent
import json
from pathlib import Path

ua = UserAgent()

def get_headers():
    """Return random user-agent headers to avoid detection."""
    return {'User-Agent': ua.random}

def rate_limit(min_wait=2, max_wait=5):
    """Random delay between requests to respect rate limits."""
    delay = random.uniform(min_wait, max_wait)
    print(f"Sleeping for {delay:.2f} seconds...")
    time.sleep(delay)


def load_config(path="config.json"):
    """Load configuration from JSON file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"❌ Config file not found at {path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

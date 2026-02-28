"""
scraper.py
Scrapes procedure information from 6 I-MED Radiology pages and saves
structured output to data/procedures.json.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": "IMED-Junior-MLE-Bot (https://github.com/DaleMYChen/imed-procedure-chatbot)"
}

REQUEST_TIMEOUT = 15          # seconds per request
DELAY_BETWEEN_REQUESTS = 1.5  # polite crawl delay (seconds)
OUTPUT_PATH = Path(__file__).parent.parent / "procedure_data" / "procedures.json"

PROCEDURE_URLS: list[dict] = [
    {"title": "Angiography",      "url": "https://i-med.com.au/procedures/angiography"},
    {"title": "Cardiac Services", "url": "https://i-med.com.au/procedures/cardiac-services"},
    {"title": "CT Scan",          "url": "https://i-med.com.au/procedures/ct-scan"},
    {"title": "General X-Ray",    "url": "https://i-med.com.au/procedures/general-x-ray"},
    {"title": "Lung Screening",   "url": "https://i-med.com.au/procedures/lung-screening"},
    {"title": "PET Scan",         "url": "https://i-med.com.au/procedures/pet-scan"},
    # {"title": "Find a Clinic", "url": "https://i-med.com.au/find-a-radiology-clinic"},

]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Procedure:
    title: str
    url: str
    raw_text: str                    # full cleaned page text (used for embedding)
    sections: dict[str, str]         # heading -> paragraph text


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------

def _fetch_html(url: str) -> Optional[str]:
    """
    Fetch raw HTML from a URL.
    Returns None on any network / HTTP error (error condition #1 handled upstream).
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()          # raises for 4xx / 5xx
        return response.text

    except requests.exceptions.ConnectionError:
        logger.error("Connection error — could not reach %s", url)
    except requests.exceptions.Timeout:
        logger.error("Request timed out after %ds for %s", REQUEST_TIMEOUT, url)
    except requests.exceptions.HTTPError as exc:
        logger.error("HTTP %s for %s", exc.response.status_code, url)
    except requests.exceptions.RequestException as exc:
        logger.error("Unexpected request error for %s: %s", url, exc)

    return None


def _parse_procedure(html: str, title: str, url: str) -> Procedure:
    """
    Parse the HTML of a procedure page into a structured Procedure object.

    I-MED pages use a consistent layout:
      - Main content lives inside <main> or a prominent <article> / <div class="...content...">
      - Headings (h1–h3) label sections; following <p> tags are the body text.

    Strategy:
      1. Extract all text for the raw_text field (used for semantic embeddings).
      2. Walk heading → paragraph pairs to build the sections dict.
    """
    soup = BeautifulSoup(html, "html.parser")

    # --- Remove noise (nav, footer, scripts, cookie banners) ---------------
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "noscript", "aside", "form"]):
        tag.decompose()

    # --- Locate main content container -------------------------------------
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_=lambda c: c and "content" in c.lower())
        or soup.body
    )

    # --- Build sections dict -----------------------------------------------
    sections: dict[str, str] = {}
    current_heading = "Overview"
    buffer: list[str] = []

    def _flush():
        text = " ".join(buffer).strip()
        if text:
            # Append to existing section text if heading already seen
            sections[current_heading] = (
                (sections[current_heading] + " " + text).strip()
                if current_heading in sections
                else text
            )
        buffer.clear()

    for element in main.descendants:
        if element.name in ("h1", "h2", "h3"):
            _flush()
            current_heading = element.get_text(separator=" ", strip=True)
        elif element.name == "p":
            text = element.get_text(separator=" ", strip=True)
            if text:
                buffer.append(text)
        elif element.name in ("li",):
            text = element.get_text(separator=" ", strip=True)
            if text:
                buffer.append("• " + text)

    _flush()  # flush final buffer

    # --- Build raw_text (flat, newline-separated for embeddings) -----------
    raw_parts = []
    for heading, body in sections.items():
        raw_parts.append(f"{heading}\n{body}")
    raw_text = "\n\n".join(raw_parts) if raw_parts else main.get_text(separator="\n", strip=True)

    return Procedure(title=title, url=url, raw_text=raw_text, sections=sections)


# ---------------------------------------------------------------------------
# Main scrape routine
# ---------------------------------------------------------------------------

def scrape_all() -> list[dict]:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    total = len(PROCEDURE_URLS)

    for i, entry in enumerate(PROCEDURE_URLS, start=1):
        title, url = entry["title"], entry["url"]
        logger.info("(%d/%d) Scraping: %s", i, total, url)

        html = _fetch_html(url)

        if html is None:
            # Error condition #1 — fetch failed; record stub so downstream
            # code can surface a clear "content unavailable" message.
            logger.warning("Skipping %s — no HTML retrieved.", title)
            results.append({
                "title": title,
                "url": url,
                "error": "Failed to retrieve page content.",
                "raw_text": "",
                "sections": {},
            })
        else:
            procedure = _parse_procedure(html, title, url)
            data = asdict(procedure)

            # Sanity check — warn if almost nothing was extracted
            if len(procedure.raw_text) < 100:
                logger.warning(
                    "%s: very little text extracted (%d chars) — "
                    "page structure may have changed.",
                    title, len(procedure.raw_text),
                )
            else:
                logger.info(
                    "%s: extracted %d chars across %d sections.",
                    title, len(procedure.raw_text), len(procedure.sections),
                )

            results.append(data)

        # Polite delay between requests (skip after last)
        if i < total:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    # --- Persist to JSON ---------------------------------------------------
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d procedures → %s", len(results), OUTPUT_PATH)
    return results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scrape_all()
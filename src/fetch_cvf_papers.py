import argparse
import logging
import os
import random
import re
import time

import pandas as pd
from playwright._impl._errors import Error as PlaywrightError
from playwright.sync_api import sync_playwright

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

parser = argparse.ArgumentParser(description="Collect missing papers from CVF.")
parser.add_argument(
    "--csv_file",
    type=str,
    default="./cvpr2024.csv",
    help="Path to the CSV file to save the collected papers.",
)
parser.add_argument(
    "--conference_url",
    type=str,
    default="https://openaccess.thecvf.com/CVPR2024?day=all",
    help="URL of the conference website to scrape.",
)
parser.add_argument(
    "--headless",
    action="store_true",
    help="Run the browser in headless mode.",
)

args = parser.parse_args()

# Constants
CSV_FILE = args.csv_file
CONFERENCE_URL = args.conference_url
MAX_RETRIES = 3
RETRY_DELAY = 3  # 에러 발생 시 재시도 간 딜레이 (초)


def preprocess_title(title: str) -> str:
    """
    논문 제목을 소문자로 변환하고 불필요한 문자를 제거하여 반환합니다.
    """
    return re.sub(r"[^a-zA-Z0-9]", "", title).lower()


def load_existing_data(csv_file: str):
    """
    CSV 파일에서 기존 데이터를 로드합니다.
    파일이 없으면 빈 DataFrame을 반환합니다.
    """
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        existing_titles = set(preprocess_title(title) for title in df["title"])
        logging.info(
            f"Loaded existing data from '{csv_file}'. Found {len(existing_titles)} papers."
        )
        return df, existing_titles
    else:
        logging.info(f"'{csv_file}' not found. Creating a new DataFrame.")
        return pd.DataFrame(columns=["title", "abstract"]), set()


def get_paper_details(page, url: str):
    """
    주어진 URL에서 논문 제목과 초록을 추출합니다.
    """
    try:
        page.goto(url)
        title = page.locator("#papertitle").inner_text()
        abstract = page.locator("#abstract").inner_text()
        return title, abstract
    except PlaywrightError as e:
        if "Locator.inner_text: Timeout" in str(e):
            logging.warning(
                f"Timeout error while getting paper details for '{url}'. Skipping paper."
            )
            return None, None
        else:
            logging.error(f"Error while getting paper details for '{url}': {e}")
            return None, None
    except Exception as e:
        logging.error(f"Error while getting paper details for '{url}': {e}")
        return None, None


def fetch_paper_details(page, title, link, existing_titles: set):
    """
    개별 논문 링크를 기반으로 정보를 가져오고, 이미 존재하는 논문인지 확인합니다.
    """
    processed_title = preprocess_title(title)

    if processed_title in existing_titles:
        logging.info(f"'{title}' already exists. Skipping...")
        return None

    url = "https://openaccess.thecvf.com" + link
    logging.info(f"Fetching details for: {title}")
    retries = 0

    while retries < MAX_RETRIES:
        try:
            paper_title, abstract = get_paper_details(page, url)
            if paper_title is not None and abstract is not None:
                logging.info(f"Successfully fetched details for: {title}")
                return {"title": paper_title, "abstract": abstract}
            else:
                return None
        except Exception as e:
            retries += 1
            logging.error(
                f"Error fetching details for '{title}' (Retry {retries}/{MAX_RETRIES}): {e}"
            )
            time.sleep(RETRY_DELAY)

    logging.error(f"Failed to fetch details for '{title}' after {MAX_RETRIES} retries.")
    return None


def main():
    """
    CVPR 홈페이지에서 누락된 논문 정보를 개별적으로 확인하고 수집합니다.
    """
    cvpr_df, existing_titles = load_existing_data(CSV_FILE)
    failed_papers = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        page = browser.new_page()
        page.goto(CONFERENCE_URL)
        logging.info(f"Navigated to {CONFERENCE_URL}")

        for _ in range(MAX_RETRIES):  # 모든 논문을 처리할 때까지 반복 시도
            paper_links = {
                a.inner_text(): a.get_attribute("href")
                for a in page.query_selector_all("dl dt a")
            }
            total_links = len(paper_links)
            logging.info(f"Found {total_links} paper links on the CVPR website.")

            for i, (title, url) in enumerate(paper_links.items()):
                try:
                    processed_title = preprocess_title(title)
                    logging.info(f"[{i+1}/{total_links}] Processing paper: '{title}'")

                    if processed_title not in existing_titles:
                        paper_info = fetch_paper_details(
                            page, title, url, existing_titles
                        )
                        if paper_info is not None:
                            new_df = pd.DataFrame([paper_info])
                            cvpr_df = pd.concat([cvpr_df, new_df], ignore_index=True)
                            existing_titles.add(preprocess_title(paper_info["title"]))
                            cvpr_df.to_csv(CSV_FILE, index=False)
                            logging.info(f"Added and saved: {paper_info['title']}")
                            logging.info("=" * 50)
                        else:
                            failed_papers.append((title, None))

                        time.sleep(5 + random.random() * 5)
                    else:
                        logging.info(f"'{title}' already exists. Skipping...")
                except Exception as e:
                    logging.error(f"Error processing paper '{title}': {e}")
                    failed_papers.append((title, e))

        browser.close()
        logging.info("Browser closed.")

    if failed_papers:
        logging.warning("\nFailed to collect the following papers:")
        for title, error in failed_papers:
            logging.warning(f"- {title} :\n {error}")
    else:
        logging.info("Successfully collected all new papers.")


if __name__ == "__main__":
    main()

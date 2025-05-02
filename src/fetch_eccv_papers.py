import argparse
import random
import re
import time

import pandas as pd
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description="Collect missing papers from ECCV.")
parser.add_argument(
    "--csv_file",
    type=str,
    default="./eccv2024.csv",
    help="Path to the CSV file to save the collected papers.",
)
args = parser.parse_args()


def get_paper_titles_and_urls(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)

        # 천천히 아래로 스크롤 (랜덤 시간 적용)
        page.evaluate("""
            async () => {
                const scrollHeight = document.documentElement.scrollHeight;
                const clientHeight = document.documentElement.clientHeight;
                const numberOfScrolls = 100;
                const scrollStep = scrollHeight / numberOfScrolls;
                for (let i = 1; i <= numberOfScrolls; i++) {
                    window.scrollTo(0, i * scrollStep);
                    const randomDelay = Math.random() * 800 + 200;
                    await new Promise(resolve => setTimeout(resolve, randomDelay));
                    if (i => 20) {
                        break;
                    }
                }
            }
        """)
        # "Click to Fetch All" 버튼 찾기
        fetch_all_button = page.locator("#btn_fetchall")

        # "Click to Fetch All" 버튼 클릭
        fetch_all_button.click()

        time.sleep(5)

        # 데이터 로딩을 기다림 (더 명시적인 방법)
        page.wait_for_selector("#paperlist tbody tr", state="visible")

        paper_data = []
        rows = page.locator("#paperlist tbody tr").all()
        for row in rows:
            title_element = row.locator("td:nth-child(2) a")
            title = title_element.inner_text()
            url = title_element.get_attribute("href")
            paper_data.append({"title": title, "url": url})

        browser.close()
        return paper_data


def get_paper_details(paper_list):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        for paper in paper_list:
            print(f"Fetching details for: {paper['title']}")
            try:
                page.goto(paper["url"])
                # 논문 상세 페이지 로딩 후 잠시 대기 (필요에 따라 조정)
                time.sleep(5 + random.random() * 5)

                abstract_element = page.locator("#abstractExample")
                abstract_text = abstract_element.inner_text()
                # Abstract: 제거, 공백 두개 이상 제거, 개행문자 제거
                abstract_text = re.sub(r"\s{2,}", " ", abstract_text)
                abstract_text = abstract_text.replace("Abstract:", "").replace(
                    "\n", " "
                )
                print(abstract_text)
                paper["abstract"] = abstract_text
                print("Abstract fetched successfully.")
            except Exception as e:
                print(f"Error fetching abstract for {paper['title']}: {e}")
                paper["abstract"] = "Abstract not found"  # 또는 None 등으로 처리

        browser.close()
        return paper_list


if __name__ == "__main__":
    target_url = (
        "https://papercopilot.com/paper-list/eccv-paper-list/eccv-2024-paper-list/"
    )
    papers = get_paper_titles_and_urls(target_url)
    print(f"Total papers found: {len(papers)}")

    papers_with_abstracts = get_paper_details(papers)

    # Create a pandas DataFrame
    df = pd.DataFrame(papers_with_abstracts)

    # Save to CSV, including only 'title' and 'abstract' columns
    df[["title", "abstract"]].to_csv(args.csv_file, index=False, mode="w")
    print("Data saved to papers_with_abstracts.csv")

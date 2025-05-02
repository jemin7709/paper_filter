import argparse
import os
import random
import time

import pandas as pd
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description="Collect missing papers from AAAI.")
parser.add_argument(
    "--path",
    type=str,
    default="./aaai2024.csv",
    help="Path to the CSV file to save the collected papers.",
)
args = parser.parse_args()


def scrape_aaai_24_papers():
    archive_url = "https://ojs.aaai.org/index.php/AAAI/issue/archive"
    data_path = args.path

    # Ensure the directory exists
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(archive_url)
        time.sleep(2)  # 페이지 로딩 대기

        # AAAI-24 Technical Tracks 링크 찾기 및 클릭
        aaai_24_link_locator = page.locator(
            '//a[contains(text(), "AAAI-24 Technical Tracks")]'
        )
        aaai_24_link_count = aaai_24_link_locator.count()
        if aaai_24_link_count > 0:
            print(
                f"{aaai_24_link_count}개의 AAAI-24 Technical Tracks 링크를 찾았습니다."
            )
            for i in range(aaai_24_link_count):
                aaai_24_link = aaai_24_link_locator.nth(i)
                if aaai_24_link:
                    aaai_24_tracks_url = aaai_24_link.get_attribute("href")
                    print(
                        f"\n--- {i+1}/{aaai_24_link_count} 번째 AAAI-24 Technical Tracks URL: {aaai_24_tracks_url} ---"
                    )
                    page.goto(aaai_24_tracks_url)
                    sleep_time = random.uniform(3, 5)
                    time.sleep(sleep_time)  # 페이지 로딩 대기

                    # 논문 목록 링크 추출
                    paper_link_elements = page.locator(
                        "ul.cmp_article_list.articles > li div.obj_article_summary h3.title > a"
                    ).all()
                    paper_urls = [
                        link.get_attribute("href") for link in paper_link_elements
                    ]

                    print(f"총 {len(paper_urls)}개의 논문 링크를 찾았습니다.")

                    for j, paper_url in enumerate(paper_urls):
                        print(
                            f"\n--- {j+1}/{len(paper_urls)} 번째 논문 처리 중: {paper_url} ---"
                        )

                        # Check if the paper already exists in the CSV
                        try:
                            existing_df = pd.read_csv(data_path)
                            if paper_url in existing_df["url"].values:
                                print(f"이미 저장된 논문입니다: {paper_url}")
                                continue
                        except FileNotFoundError:
                            pass  # It's okay if the file doesn't exist yet

                        try:
                            page.goto(paper_url)
                            sleep_time = random.uniform(3, 5)
                            time.sleep(sleep_time)  # 페이지 로딩 대기

                            # 제목 추출
                            title_element = page.query_selector("h1.page_title")
                            if title_element:
                                title = title_element.text_content().strip()
                                print(f"제목: {title}")
                            else:
                                title = "제목을 찾을 수 없습니다."
                                print(title)

                            # 초록 추출
                            abstract_element = page.query_selector(
                                "section.item.abstract"
                            )
                            if abstract_element:
                                abstract_text_element = abstract_element.query_selector(
                                    "div"
                                )
                                if abstract_text_element:
                                    abstract = (
                                        abstract_text_element.text_content().strip()
                                    )
                                    print(f"초록: {abstract}")
                                else:
                                    # div가 없는 경우, section.item.abstract 바로 아래 텍스트 노드를 시도
                                    abstract = abstract_element.inner_text().strip()
                                    if abstract.startswith(
                                        "Abstract"
                                    ):  # "Abstract" 라는 단어로 시작하는지 확인
                                        abstract = abstract[
                                            len("Abstract") :
                                        ].strip()  # "Abstract" 제거
                                        print(f"초록: {abstract}")
                                    else:
                                        abstract = "초록 텍스트를 찾을 수 없습니다 (텍스트 노드)."
                                        print(abstract)

                            else:
                                abstract = "초록 섹션을 찾을 수 없습니다."
                                print(abstract)

                            paper_data_dict = {
                                "title": [title],
                                "abstract": [abstract],
                                "url": [paper_url],
                            }
                            paper_df = pd.DataFrame(paper_data_dict)

                            # Save to CSV, appending and avoiding header if file exists
                            if not os.path.exists(data_path):
                                paper_df.to_csv(
                                    data_path, mode="a", header=True, index=False
                                )
                            else:
                                paper_df.to_csv(
                                    data_path, mode="a", header=False, index=False
                                )

                            sleep_time = random.uniform(5, 10)  # 5~10초 사이 랜덤 sleep
                            time.sleep(sleep_time)

                        except Exception as e:
                            print(f"오류 발생: {e}")
                else:
                    print(
                        f"{i+1}번째 AAAI-24 Technical Tracks 링크 요소를 찾았지만, 가져오는 데 실패했습니다."
                    )
                page.goto(archive_url)
                time.sleep(2)
        else:
            print("AAAI-24 Technical Tracks 링크를 찾을 수 없습니다.")

        browser.close()

    print("\n--- AAAI-24 Technical Tracks 논문 스크래핑 완료 ---")


if __name__ == "__main__":
    scrape_aaai_24_papers()
    print("결과는 ./data/aaai2024.csv 파일에 저장되었습니다.")

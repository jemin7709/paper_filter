import argparse
import os
import random
import re
import time
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from playwright.sync_api import Page, sync_playwright


def parse_arguments():
    parser = argparse.ArgumentParser(description="ECCV 논문 수집")
    parser.add_argument(
        "--year",
        type=str,
        default="24",
        help="학회 연도 (ex: 24)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./data/eccv.csv",
        help="저장할 CSV 파일 경로",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="브라우저 창을 보이지 않게 설정",
    )
    parser.add_argument(
        "--proxy",
        action="store_true",
        help="Tor 사용하는 경우 체크",
    )
    return parser.parse_args()


def get_proxy_settings() -> Dict[str, str]:
    SOCKS_PROXY_ADDRESS = "127.0.0.1"
    SOCKS_PROXY_PORT = 9050
    return {"server": f"socks5://{SOCKS_PROXY_ADDRESS}:{SOCKS_PROXY_PORT}"}


def preprocess_title(title: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", title).lower()


def load_existing_data(csv_file: str) -> Tuple[pd.DataFrame, Set[str]]:
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        existing_titles = set(preprocess_title(title) for title in df["title"])
        print(
            f"'{csv_file}'에서 기존 데이터를 로드했습니다. {len(existing_titles)}개의 논문을 찾았습니다."
        )
        return df, existing_titles
    else:
        print(f"'{csv_file}'을 찾을 수 없습니다. 새 DataFrame을 생성합니다.")
        return pd.DataFrame(columns=["title", "abstract", "url"]), set()


def get_paper_titles_and_urls(
    url: str, headless: bool = True, proxy: Dict[str, str] = None
) -> List[Dict[str, str]]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, proxy=proxy)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded")

        view_height = page.viewport_size["height"]
        current_position = 0

        for _ in range(20):
            current_position += int(view_height * 0.8)
            page.mouse.wheel(0, view_height)
            delay = random.uniform(0.3, 0.5)
            time.sleep(delay)

        fetch_all_button = page.locator("#btn_fetchall")
        fetch_all_button.click()

        time.sleep(random.uniform(3, 5))
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


def get_abstract(page: Page, paper: Dict[str, str]) -> Optional[str]:
    try:
        abstract_element = page.locator("#abstractExample")
        abstract_text = abstract_element.inner_text()
        abstract_text = re.sub(r"\s{2,}", " ", abstract_text)
        abstract_text = (
            abstract_text.replace("Abstract:", "").replace("\n", " ").strip()
        )
        return abstract_text
    except Exception as e:
        print(f"Abstract 가져오기 오류: {e}")
        return None


def get_paper_details(
    paper_list: List[Dict[str, str]],
    existing_titles: Set[str],
    df: pd.DataFrame,
    csv_file: str,
    headless: bool = False,
    proxy: Dict[str, str] = None,
) -> Tuple[pd.DataFrame, Set[str], List[Tuple[str, Optional[Exception]]]]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, proxy=proxy)
        page = browser.new_page()
        failed_papers = []

        for i, paper in enumerate(paper_list):
            print(f"[{i + 1}/{len(paper_list)}] 번째 논문 처리 중")

            processed_title = preprocess_title(paper["title"])
            if processed_title in existing_titles:
                print(f"'{paper['title']}' 논문은 이미 존재합니다. 건너뜁니다...")
                print("=" * 50)
                continue

            MAX_RETRIES = 3
            retries = 0

            while retries < MAX_RETRIES:
                try:
                    page.goto(paper["url"])
                    time.sleep(random.uniform(3, 5))

                    abstract_text = get_abstract(page, paper)
                    if abstract_text:
                        paper["abstract"] = abstract_text

                        # 결과를 바로 CSV에 저장
                        df = save_paper_to_csv(paper, df, csv_file)
                        existing_titles.add(processed_title)
                        break
                    else:
                        paper["abstract"] = "Abstract를 찾을 수 없음"
                        print("Abstract를 찾을 수 없습니다.")
                        failed_papers.append((paper["title"], None))
                        break
                except Exception as e:
                    retries += 1
                    print(
                        f"Abstract 가져오기 오류 (재시도 {retries}/{MAX_RETRIES}): {e}"
                    )
                    if retries >= MAX_RETRIES:
                        paper["abstract"] = "Abstract 가져오기 오류"
                        failed_papers.append((paper["title"], e))
                    time.sleep(random.uniform(3, 5))

            print("=" * 50)
            time.sleep(random.uniform(3, 5))

        browser.close()
        return df, existing_titles, failed_papers


def save_paper_to_csv(
    paper: Dict[str, str], df: pd.DataFrame, csv_file: str
) -> pd.DataFrame:
    new_row = pd.DataFrame(
        [{"title": paper["title"], "abstract": paper["abstract"], "url": paper["url"]}]
    )
    updated_df = pd.concat([df, new_row], ignore_index=True)
    updated_df.to_csv(csv_file, index=False)
    print("논문 추가 및 저장 완료")
    print(f"Title: {paper['title']}")
    print(f"Abstract: {paper['abstract'][:100]}...")
    return updated_df


def scrape_eccv_papers(
    year: str, csv_file: str, headless: bool = False, use_proxy: bool = False
) -> None:
    target_url = (
        f"https://papercopilot.com/paper-list/eccv-paper-list/eccv-20{year}-paper-list/"
    )

    # CSV 파일이 저장될 디렉토리 확인
    output_dir = os.path.dirname(csv_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 기존 데이터 로드
    df, existing_titles = load_existing_data(csv_file)

    # 프록시 설정
    proxy = get_proxy_settings() if use_proxy else None

    # 논문 목록 가져오기
    papers = get_paper_titles_and_urls(target_url, headless, proxy)
    print(f"총 {len(papers)}개의 논문을 찾았습니다.")

    # 논문 상세 정보 가져오기
    df, existing_titles, failed_papers = get_paper_details(
        papers, existing_titles, df, csv_file, headless, proxy
    )

    # 실패한 논문 정보 출력
    if failed_papers:
        print("\n아래 논문을 스크래핑하지 못했습니다:")
        for title, error in failed_papers:
            print(f"- {title} :\n {error}")
    else:
        print("모든 논문을 성공적으로 스크래핑했습니다.")

    print(f"\n--- ECCV 20{year} 논문 스크래핑 완료 ---")
    print(f"스크래핑 결과를 {csv_file} 파일에 저장했습니다.")


def main():
    args = parse_arguments()
    csv_file = args.path.replace("eccv.csv", f"eccv_{args.year}.csv")

    scrape_eccv_papers(
        year=args.year,
        csv_file=csv_file,
        headless=args.headless,
        use_proxy=args.proxy,
    )


if __name__ == "__main__":
    main()

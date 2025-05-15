import argparse
import os
import random
import re
import time
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from playwright._impl._errors import Error as PlaywrightError
from playwright.sync_api import Page, sync_playwright


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="CVF 웹사이트에서 논문 정보를 수집합니다."
    )
    parser.add_argument(
        "--year",
        type=str,
        default="24",
        help="학회 연도 (ex: 24)",
    )
    parser.add_argument(
        "--conference",
        type=str,
        default="CVPR",
        help="학회 이름 (ex: CVPR, ICCV, WACV)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./data/cvpr.csv",
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


def get_paper_details(page: Page, url: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        page.goto(url)
        title = page.locator("#papertitle").inner_text()
        abstract = page.locator("#abstract").inner_text()
        return title, abstract
    except PlaywrightError as e:
        if "Locator.inner_text: Timeout" in str(e):
            print(f"'{url}'에서 논문 상세정보를 가져오는 중 타임아웃 발생. 건너뜁니다.")
            return None, None
        else:
            print(f"'{url}'에서 논문 상세정보를 가져오는 중 오류 발생: {e}")
            return None, None
    except Exception as e:
        print(f"'{url}'에서 논문 상세정보를 가져오는 중 오류 발생: {e}")
        return None, None


def fetch_paper_details(
    page: Page, title: str, link: str, existing_titles: Set[str]
) -> Optional[Dict[str, str]]:
    MAX_RETRIES = 3
    RETRY_DELAY = 3

    processed_title = preprocess_title(title)

    if processed_title in existing_titles:
        print(f"'{title}' 논문은 이미 존재합니다. 건너뜁니다...")
        return None

    full_url = "https://openaccess.thecvf.com" + link
    retries = 0

    while retries < MAX_RETRIES:
        try:
            paper_title, abstract = get_paper_details(page, full_url)
            if paper_title is not None and abstract is not None:
                return {"title": paper_title, "abstract": abstract, "url": full_url}
            else:
                return None
        except Exception as e:
            retries += 1
            print(
                f"'{title}' 논문 상세정보 가져오기 실패 (재시도 {retries}/{MAX_RETRIES}): {e}"
            )
            time.sleep(RETRY_DELAY)

    print(f"'{title}' 논문 상세정보 가져오기를 {MAX_RETRIES}회 재시도 후 실패했습니다.")
    return None


def save_to_csv(
    paper_data: Dict[str, str], df: pd.DataFrame, csv_file: str
) -> pd.DataFrame:
    new_df = pd.DataFrame([paper_data])
    updated_df = pd.concat([df, new_df], ignore_index=True)
    updated_df.to_csv(csv_file, index=False)
    print(f"논문 추가 및 저장 완료: {paper_data['title']}")
    return updated_df


def get_paper_links(page: Page) -> Dict[str, str]:
    paper_links = {
        a.inner_text(): a.get_attribute("href")
        for a in page.query_selector_all("dl dt a")
    }
    total_links = len(paper_links)
    print(f"CVF 웹사이트에서 {total_links}개의 논문 링크를 찾았습니다.")
    return paper_links


def process_paper_links(
    page: Page,
    paper_links: Dict[str, str],
    existing_titles: Set[str],
    cvpr_df: pd.DataFrame,
    csv_file: str,
) -> Tuple[pd.DataFrame, Set[str], List[Tuple[str, Optional[Exception]]]]:
    failed_papers = []
    total_links = len(paper_links)

    for i, (title, url) in enumerate(paper_links.items()):
        try:
            processed_title = preprocess_title(title)
            print(f"[{i + 1}/{total_links}] 번째 논문 처리 중")

            if processed_title not in existing_titles:
                paper_info = fetch_paper_details(page, title, url, existing_titles)
                if paper_info is not None:
                    cvpr_df = save_to_csv(paper_info, cvpr_df, csv_file)
                    existing_titles.add(preprocess_title(paper_info["title"]))
                    print("=" * 50)
                else:
                    failed_papers.append((title, None))

                time.sleep(random.uniform(3, 5))
            else:
                print(f"'{title}' 논문은 이미 존재합니다. 건너뜁니다...")
        except Exception as e:
            print(f"'{title}' 논문 처리 중 오류 발생: {e}")
            failed_papers.append((title, e))

    return cvpr_df, existing_titles, failed_papers


def scrape_cvf_papers(
    conference_url: str, csv_file: str, headless: bool = False, use_proxy: bool = False
) -> None:
    MAX_RETRIES = 3
    cvpr_df, existing_titles = load_existing_data(csv_file)
    failed_papers = []

    output_dir = os.path.dirname(csv_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with sync_playwright() as p:
        proxy = get_proxy_settings() if use_proxy else None
        browser = p.chromium.launch(headless=headless, proxy=proxy)
        page = browser.new_page()
        page.goto(conference_url)
        print(f"{conference_url}로 이동했습니다.")

        for attempt in range(MAX_RETRIES):
            paper_links = get_paper_links(page)

            if not paper_links:
                print(
                    f"논문 링크를 찾을 수 없습니다. 재시도 중... ({attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(random.uniform(3, 5))
                continue

            cvpr_df, existing_titles, failed = process_paper_links(
                page, paper_links, existing_titles, cvpr_df, csv_file
            )
            failed_papers.extend(failed)
            break

        browser.close()
        print("브라우저를 종료했습니다.")

    if failed_papers:
        print("\n다음 논문을 스크래핑하지 못했습니다:")
        for title, error in failed_papers:
            print(f"- {title} :\n {error}")
    else:
        print("모든 논문을 성공적으로 스크래핑했습니다.")


def main():
    args = parse_arguments()
    args.path = args.path.replace(
        "cvpr.csv", f"{args.conference.lower()}_{args.year}.csv"
    )
    conference_url = (
        f"https://openaccess.thecvf.com/{args.conference}20{args.year}" + "?day=all"
        if args.conference != "WACV"
        else ""
    )

    scrape_cvf_papers(
        conference_url=conference_url,
        csv_file=args.path,
        headless=args.headless,
        use_proxy=args.proxy,
    )

    print(f"스크래핑 결과를 {args.path} 파일에 저장했습니다.")


if __name__ == "__main__":
    main()

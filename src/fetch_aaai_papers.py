import argparse
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from playwright.sync_api import Page, sync_playwright


def parse_arguments():
    parser = argparse.ArgumentParser(description="AAAI 논문 스크래핑")
    parser.add_argument(
        "--year",
        type=str,
        default="25",
        help="학회 연도 (ex: 24, 25)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./data/aaai.csv",
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


def extract_paper_info(page: Page, paper_url: str) -> Dict[str, str]:
    print(f"페이지 이동: {paper_url}")
    page.goto(paper_url)
    time.sleep(random.uniform(3, 5))

    title = ""
    title_element = page.query_selector("h1.page_title")
    if title_element:
        title = title_element.text_content().strip()
        print(f"Title: {title}")

    abstract = ""
    abstract_element = page.query_selector("section.item.abstract")
    if abstract_element:
        abstract_text_element = abstract_element.query_selector("div")
        if abstract_text_element:
            abstract = abstract_text_element.text_content().strip()
        else:
            abstract = abstract_element.inner_text().strip()
            if abstract.startswith("Abstract"):
                abstract = abstract[len("Abstract") :].strip()
        print(f"Abstract: {abstract[:100]}...")

    return {
        "title": title,
        "abstract": abstract,
        "url": paper_url,
    }


def save_to_csv(paper_data: Dict[str, str], filepath: str) -> None:
    paper_df = pd.DataFrame(
        {
            "title": [paper_data["title"]],
            "abstract": [paper_data["abstract"]],
            "url": [paper_data["url"]],
        }
    )

    file_exists = os.path.exists(filepath)
    paper_df.to_csv(filepath, mode="a", header=not file_exists, index=False)


def is_paper_already_saved(paper_url: str, filepath: str) -> bool:
    try:
        if not os.path.exists(filepath):
            return False
        existing_df = pd.read_csv(filepath)
        return paper_url in existing_df["url"].values
    except Exception as e:
        print(f"CSV 파일 확인 중 오류 발생: {e}")
        return False


def extract_year_from_text(text: str) -> Optional[int]:
    aaai_pattern = re.search(r"AAAI-(\d{2})", text)
    if aaai_pattern:
        return int(aaai_pattern.group(1))

    year_pattern = re.search(r"20(\d{2})", text)
    if year_pattern:
        return int(year_pattern.group(1))

    return None


def check_track_year(track_title: str, target_year: str) -> Tuple[bool, Optional[int]]:
    target_year_int = int(target_year)
    track_year = extract_year_from_text(track_title)

    if track_year is None:
        print(f"트랙에서 연도를 추출할 수 없습니다: {track_title}")
        return True, None

    print(f"현재 트랙 연도: {track_year}, 타겟 연도: {target_year_int}")

    if track_year > target_year_int:
        print(
            f"타겟 연도({target_year_int})보다 이후 연도({track_year}) 트랙입니다. 건너뜁니다."
        )
        return False, track_year
    elif track_year < target_year_int:
        print(
            f"타겟 연도({target_year_int})보다 이전 연도({track_year}) 트랙입니다. 종료합니다."
        )
        return False, track_year
    else:
        print(f"타겟 연도({target_year_int})와 일치하는 트랙입니다. 처리합니다.")
        return True, track_year


def find_aaai_track_links(page: Page) -> List[Tuple[str, str]]:
    track_info = []
    issue_links = page.locator('//a[contains(text(), "Technical Tracks")]').all()

    if len(issue_links) > 0:
        print(f"{len(issue_links)}개의 Technical Tracks 링크를 찾았습니다.")

        for link in issue_links:
            track_title = link.text_content().strip()
            track_url = link.get_attribute("href")

            if track_url:
                track_info.append((track_title, track_url))
                print(f"트랙 정보: {track_title} | URL: {track_url}")
    else:
        print("Technical Tracks 링크를 찾을 수 없습니다.")

    return track_info


def get_paper_urls(page: Page, track_url: str) -> List[str]:
    page.goto(track_url)
    time.sleep(random.uniform(3, 5))

    paper_link_elements = page.locator(
        "ul.cmp_article_list.articles > li div.obj_article_summary h3.title > a"
    ).all()

    paper_urls = [link.get_attribute("href") for link in paper_link_elements]
    print(f"총 {len(paper_urls)}개의 논문 링크를 찾았습니다.")
    return paper_urls


def go_to_next_page(page: Page) -> bool:
    next_link = page.query_selector("a.next")

    if next_link:
        next_url = next_link.get_attribute("href")
        if next_url:
            print(f"다음 페이지로 이동: {next_url}")
            page.goto(next_url)
            time.sleep(random.uniform(3, 5))
            return True

    print("다음 페이지가 없습니다.")
    return False


def process_archive_page(page: Page, target_year: str, output_path: str) -> bool:
    track_info = find_aaai_track_links(page)

    found_target_year = False
    all_later_years = True

    for i, (track_title, track_url) in enumerate(track_info):
        print(
            f"\n--- {i + 1}/{len(track_info)} 번째 Technical Track: {track_title} ---"
        )

        should_process_track, track_year = check_track_year(track_title, target_year)

        if track_year is not None:
            if track_year < int(target_year):
                print(f"타겟 연도({target_year})보다 이전 트랙이므로 종료합니다.")
                return False
            elif track_year == int(target_year):
                found_target_year = True
                all_later_years = False

        if not should_process_track:
            if track_year is not None and track_year > int(target_year):
                print(f"타겟 연도({target_year})보다 이후 트랙이므로 건너뜁니다.")
                continue

        paper_urls = get_paper_urls(page, track_url)

        for j, paper_url in enumerate(paper_urls):
            print(f"\n--- {j + 1}/{len(paper_urls)} 번째 논문 처리 중 ---")

            if is_paper_already_saved(paper_url, output_path):
                print(f"이미 저장된 논문입니다: {paper_url}")
                continue

            try:
                MAX_RETRIES = 3
                retries = 0
                success = False

                while retries < MAX_RETRIES and not success:
                    try:
                        paper_data = extract_paper_info(page, paper_url)
                        save_to_csv(paper_data, output_path)
                        time.sleep(random.uniform(3, 5))
                        success = True
                    except Exception as e:
                        retries += 1
                        print(
                            f"논문 처리 중 오류 발생 (재시도 {retries}/{MAX_RETRIES}): {e}"
                        )
                        time.sleep(3)

                if not success:
                    print(f"{MAX_RETRIES}회 재시도 후 실패했습니다.")
            except Exception as e:
                print(f"논문 처리 중 오류 발생: {e}")

        page.goto(page.url)
        time.sleep(random.uniform(2, 3))

    if all_later_years and not found_target_year:
        print(
            "현재 페이지의 모든 트랙이 타겟 연도보다 이후입니다. 다음 페이지로 이동합니다."
        )
        return True

    if found_target_year:
        return False

    return True


def scrape_aaai_papers(
    year: str, output_path: str, headless: bool = False, use_proxy: bool = False
):
    archive_url = "https://ojs.aaai.org/index.php/AAAI/issue/archive"
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with sync_playwright() as p:
        proxy = get_proxy_settings() if use_proxy else None
        browser = p.chromium.launch(headless=headless, proxy=proxy)
        page = browser.new_page()

        try:
            page.goto(archive_url)
            time.sleep(random.uniform(3, 5))

            current_page = 1
            while True:
                print(f"\n=== 아카이브 페이지 {current_page} 처리 중 ===")

                should_continue = process_archive_page(page, year, output_path)
                if not should_continue:
                    print(
                        "적절한 연도를 찾았거나 이전 연도를 발견했습니다. 스크래핑을 종료합니다."
                    )
                    break

                has_next_page = go_to_next_page(page)
                if not has_next_page:
                    print("더 이상 다음 페이지가 없습니다. 스크래핑을 종료합니다.")
                    break

                current_page += 1

        except Exception as e:
            print(f"스크래핑 중 오류 발생: {e}")

        finally:
            browser.close()

    print(f"\n--- AAAI-{year} Technical Tracks 논문 스크래핑 완료 ---")


def main():
    args = parse_arguments()
    output_path = args.path.replace(".csv", f"_{args.year}.csv")

    scrape_aaai_papers(
        year=args.year,
        output_path=output_path,
        headless=args.headless,
        use_proxy=args.proxy,
    )

    print(f"스크래핑 결과를 {output_path} 파일에 저장했습니다.")


if __name__ == "__main__":
    main()

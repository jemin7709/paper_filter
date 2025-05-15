import argparse
import os
from typing import Dict, List

import openreview
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(description="OpenReview 논문 수집")
    parser.add_argument(
        "--year",
        type=str,
        default="24",
        help="학회 연도",
    )
    parser.add_argument(
        "--conference",
        type=str,
        default="ICLR",
        help="학회 이름 (ex: ICLR, NeurIPS)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./data/openreview.csv",
        help="저장할 CSV 파일 경로",
    )
    return parser.parse_args()


def initialize_openreview_client() -> openreview.api.OpenReviewClient:
    try:
        client = openreview.api.OpenReviewClient(
            baseurl="https://api2.openreview.net", username="", password=""
        )
        return client
    except Exception as e:
        print(f"OpenReview 클라이언트 초기화 실패: {e}")
        raise


def get_venue_submissions(
    client: openreview.api.OpenReviewClient, venue_id: str
) -> List:
    try:
        venue_group = client.get_group(venue_id)
        submission_name = venue_group.content["submission_name"]["value"]
        submissions = client.get_all_notes(
            invitation=f"{venue_id}/-/{submission_name}", details="replies"
        )
        print(f"{venue_id}에서 총 {len(submissions)}개의 제출 논문을 찾았습니다.")
        return submissions
    except Exception as e:
        print(f"학회 제출 논문 가져오기 실패: {e}")
        raise


def preprocess_venue(venue_value: str) -> str:
    if "oral" in venue_value.lower():
        return "Accepted(Oral)"
    elif "spotlight" in venue_value.lower():
        return "Accepted(Spotlight)"
    elif "poster" in venue_value.lower():
        return "Accepted(Poster)"
    elif "withdrawn" in venue_value.lower():
        return "Withdrawn"
    elif "desk rejected" in venue_value.lower():
        return "Desk Rejected"
    elif "submitted" in venue_value.lower():
        return "Rejected"
    else:
        return "None"


def submission_to_note(submission) -> Dict:
    note = {
        "title": submission.content["title"]["value"],
        "decision": preprocess_venue(submission.content["venue"]["value"]),
        "abstract": submission.content["abstract"]["value"],
        "url": f"https://openreview.net/forum?id={submission.id}",
    }
    return note


def process_submissions(submissions: List) -> pd.DataFrame:
    notes = [submission_to_note(submission) for submission in submissions]
    notes_df = pd.DataFrame(notes)
    accepted_notes = notes_df[notes_df["decision"].str.contains("Accepted")]
    print(
        f"총 {len(notes_df)}개 중 {len(accepted_notes)}개의 채택된 논문을 찾았습니다."
    )
    return accepted_notes


def save_to_csv(df: pd.DataFrame, output_file: str) -> None:
    df.drop(columns=["decision"], inplace=True)
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_file, index=False, mode="w")
    print(f"총 {len(df)}개의 논문을 {output_file}에 저장했습니다.")


def extract_conference_info(venue_id: str) -> str:
    parts = venue_id.split("/")
    if len(parts) >= 2:
        conference = parts[0].split(".")[0].lower()
        year = parts[1]
        return f"{conference}_{year}"
    return "openreview"


def fetch_openreview_papers(venue_id: str, path: str) -> None:
    try:
        client = initialize_openreview_client()
        submissions = get_venue_submissions(client, venue_id)
        notes_df = process_submissions(submissions)
        save_to_csv(notes_df, path)

    except Exception as e:
        print(f"논문 수집 중 오류 발생: {e}")


def main():
    args = parse_arguments()
    venue_id = f"{args.conference}.cc/20{args.year}/Conference"
    args.path = args.path.replace(
        "openreview.csv", f"{args.conference.lower()}_{args.year}.csv"
    )
    fetch_openreview_papers(
        venue_id=venue_id,
        path=args.path,
    )


if __name__ == "__main__":
    main()

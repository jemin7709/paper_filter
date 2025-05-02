import argparse

import openreview
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--venue_id", type=str, default="", help="ICLR.cc/2024/Conference")
parser.add_argument("--output_file", type=str, default="./iclr2024.csv")
args = parser.parse_args()

client = openreview.api.OpenReviewClient(
    baseurl="https://api2.openreview.net", username="", password=""
)


venue_id = args.venue_id
venue_group = client.get_group(venue_id)
submission_name = venue_group.content["submission_name"]["value"]
submissions = client.get_all_notes(
    invitation=f"{venue_id}/-/{submission_name}", details="replies"
)


def preprocess_venue(venue_value):
    """Extracts accept type and other status information from the venue string."""
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


def submission2note(submission):
    review_keys = ["summary", "strengths", "weaknesses", "questions"]
    total_replies = submission.details["replies"]
    rating_replies = [
        reply for reply in submission.details["replies"] if "rating" in reply["content"]
    ]
    note = {
        "id": submission.id,
        "title": submission.content["title"]["value"],
        "decision": preprocess_venue(submission.content["venue"]["value"]),
        # "keywords": submission.content["keywords"]["value"],
        "ratings": [
            int(reply["content"]["rating"]["value"]) for reply in rating_replies
        ],
        "confidences": [
            int(reply["content"]["confidence"]["value"]) for reply in rating_replies
        ],
        "withdraw": 1 if "Withdrawn" in submission.content["venue"]["value"] else 0,
        "review_lengths": [
            sum([len(reply["content"][key]["value"].split()) for key in review_keys])
            for reply in rating_replies
        ],
        "abstract": submission.content["abstract"]["value"],
        "comments": len(total_replies),
        "url": f"https://openreview.net/forum?id={submission.id}",
    }
    return note


notes = [submission2note(submission) for submission in submissions]
notes = pd.DataFrame(notes)
notes = notes[notes["decision"].str.contains("Accepted")]
# notes["ratings_avg"] = notes["ratings"].apply(lambda x: np.mean(x))
# notes["ratings_std"] = notes["ratings"].apply(lambda x: np.std(x))
# notes["confidence_avg"] = notes["confidences"].apply(lambda x: np.mean(x))
# notes["confidence_std"] = notes["confidences"].apply(lambda x: np.std(x))

notes.to_csv(f"{args.output_file}", index=False, mode="w")

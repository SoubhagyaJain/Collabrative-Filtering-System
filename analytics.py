import json
from pathlib import Path

import pandas as pd


def load_events(log_path="logs/events.jsonl"):
    path = Path(log_path)
    if not path.exists():
        return pd.DataFrame()

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def summarize_experiment(log_path="logs/events.jsonl"):
    events_df = load_events(log_path)
    if events_df.empty:
        return pd.DataFrame(columns=["variant", "impressions", "clicks", "likes", "ctr", "like_rate"])

    impressions = events_df[events_df["event_type"] == "impression"].groupby("variant").size().rename("impressions")
    clicks = events_df[events_df["event_type"] == "click"].groupby("variant").size().rename("clicks")
    likes = events_df[events_df["event_type"] == "like"].groupby("variant").size().rename("likes")

    summary = pd.concat([impressions, clicks, likes], axis=1).fillna(0)
    summary["ctr"] = summary["clicks"] / summary["impressions"].replace(0, 1)
    summary["like_rate"] = summary["likes"] / summary["impressions"].replace(0, 1)

    return summary.reset_index()

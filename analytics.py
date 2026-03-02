from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


IMPRESSION_EVENTS = {"impression"}
CLICK_EVENTS = {"click"}
ENGAGEMENT_EVENTS = {"like", "watch_start"}


def load_event_logs(path: str | Path) -> pd.DataFrame:
    """Load event logs from CSV and normalize expected columns."""
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()

    logs = pd.read_csv(file_path)
    if logs.empty:
        return logs

    required = {"variant", "event_type"}
    missing = required.difference(logs.columns)
    if missing:
        raise ValueError(f"Missing required columns in event logs: {sorted(missing)}")

    logs = logs.copy()
    logs["variant"] = logs["variant"].astype(str)
    logs["event_type"] = logs["event_type"].astype(str).str.lower().str.strip()

    if "rank" in logs.columns:
        logs["rank"] = pd.to_numeric(logs["rank"], errors="coerce")

    return logs


def _binomial_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for binomial proportions."""
    if total <= 0:
        return np.nan, np.nan

    p_hat = successes / total
    denominator = 1 + (z**2 / total)
    center = (p_hat + (z**2 / (2 * total))) / denominator
    margin = (
        z
        * np.sqrt((p_hat * (1 - p_hat) / total) + ((z**2) / (4 * (total**2))))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def compute_variant_metrics(logs: pd.DataFrame, baseline_variant: Optional[str] = None) -> pd.DataFrame:
    """Compute per-variant product metrics from event logs."""
    if logs.empty:
        return pd.DataFrame()

    rows = []
    for variant, frame in logs.groupby("variant"):
        impressions = int(frame["event_type"].isin(IMPRESSION_EVENTS).sum())
        clicks = int(frame["event_type"].isin(CLICK_EVENTS).sum())
        engagements = int(frame["event_type"].isin(ENGAGEMENT_EVENTS).sum())

        ctr = (clicks / impressions) if impressions else np.nan
        engagement_rate = (engagements / impressions) if impressions else np.nan

        clicked_rows = frame[frame["event_type"].isin(CLICK_EVENTS)]
        avg_rank_clicked = (
            clicked_rows["rank"].dropna().mean()
            if "rank" in clicked_rows.columns and not clicked_rows.empty
            else np.nan
        )

        ctr_lo, ctr_hi = _binomial_ci(clicks, impressions)
        eng_lo, eng_hi = _binomial_ci(engagements, impressions)

        rows.append(
            {
                "variant": variant,
                "impressions": impressions,
                "clicks": clicks,
                "engagements": engagements,
                "ctr": ctr,
                "ctr_ci_low": ctr_lo,
                "ctr_ci_high": ctr_hi,
                "engagement_rate": engagement_rate,
                "engagement_ci_low": eng_lo,
                "engagement_ci_high": eng_hi,
                "avg_rank_clicked": avg_rank_clicked,
            }
        )

    metrics = pd.DataFrame(rows).sort_values("variant").reset_index(drop=True)

    if baseline_variant is None and not metrics.empty:
        baseline_variant = metrics.loc[0, "variant"]

    if baseline_variant in set(metrics["variant"]):
        baseline_ctr = float(metrics.loc[metrics["variant"] == baseline_variant, "ctr"].iloc[0])
        metrics["baseline_variant"] = baseline_variant
        if pd.notna(baseline_ctr) and baseline_ctr > 0:
            metrics["ctr_uplift_vs_baseline"] = (metrics["ctr"] - baseline_ctr) / baseline_ctr
        else:
            metrics["ctr_uplift_vs_baseline"] = np.nan
    else:
        metrics["baseline_variant"] = np.nan
        metrics["ctr_uplift_vs_baseline"] = np.nan

    return metrics

import hashlib
from datetime import date

EXPERIMENT_NAME = "ranking_pipeline_ab_test"
EXPERIMENT_START_DATE = date(2026, 1, 1)
EXPERIMENT_TRAFFIC_SPLIT = {
    "control": 0.5,
    "treatment": 0.5,
}
EXPERIMENT_SUCCESS_METRICS = [
    "ctr_recommendation_click",
    "conversion_watch_started",
    "avg_session_watch_time_minutes",
]


def assign_variant(user_id, experiment_name, traffic_split):
    """Deterministically assign a user to an experiment variant using hashing."""
    if not traffic_split:
        raise ValueError("traffic_split cannot be empty")

    total = sum(traffic_split.values())
    if total <= 0:
        raise ValueError("traffic_split must have a positive total")

    normalized_split = {
        variant: weight / total
        for variant, weight in traffic_split.items()
    }

    bucket_source = f"{experiment_name}:{user_id}".encode("utf-8")
    bucket_int = int(hashlib.sha256(bucket_source).hexdigest(), 16)
    bucket_value = (bucket_int % 10_000) / 10_000

    running_threshold = 0.0
    selected_variant = None
    for variant, ratio in normalized_split.items():
        running_threshold += ratio
        if bucket_value < running_threshold:
            selected_variant = variant
            break

    # Guard against edge-case floating point issues
    if selected_variant is None:
        selected_variant = next(reversed(normalized_split))

    return selected_variant

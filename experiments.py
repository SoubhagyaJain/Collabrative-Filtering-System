import hashlib


EXPERIMENT_NAME = "ranking_v1"
TRAFFIC_SPLIT = {"control": 0.5, "treatment": 0.5}


def assign_variant(user_id, experiment_name=EXPERIMENT_NAME, traffic_split=TRAFFIC_SPLIT):
    """Deterministically assign a user to an experiment variant."""
    seed = f"{experiment_name}:{user_id}".encode("utf-8")
    bucket = int(hashlib.md5(seed).hexdigest(), 16) % 10000 / 10000

    cumulative = 0.0
    for variant, ratio in traffic_split.items():
        cumulative += ratio
        if bucket < cumulative:
            return variant
    return list(traffic_split.keys())[-1]

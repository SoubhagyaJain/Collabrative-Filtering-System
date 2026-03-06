import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


COLUMN_NAMES = ["User_ID", "User_Names", "Movie_ID", "Rating", "Timestamp"]


@dataclass
class EvalResult:
    variant: str
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    catalog_coverage: float
    users_evaluated: int
    interactions_train: int
    interactions_test: int
    k: int
    leave_last_n: int


def load_ratings(path: str) -> pd.DataFrame:
    """Load ratings data and normalize dtypes."""
    ratings = pd.read_csv(path, names=COLUMN_NAMES)
    ratings["Timestamp"] = pd.to_numeric(ratings["Timestamp"], errors="coerce")
    ratings["Rating"] = pd.to_numeric(ratings["Rating"], errors="coerce")
    ratings = ratings.dropna(subset=["User_ID", "Movie_ID", "Rating", "Timestamp"])
    ratings["User_ID"] = ratings["User_ID"].astype(int)
    ratings["Movie_ID"] = ratings["Movie_ID"].astype(int)
    ratings["Rating"] = ratings["Rating"].astype(float)
    ratings["Timestamp"] = ratings["Timestamp"].astype(int)
    return ratings


def leave_last_n_split(ratings: pd.DataFrame, leave_last_n: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split: hold out each user's last N interactions for test."""
    sorted_ratings = ratings.sort_values(["User_ID", "Timestamp"]) 
    split_parts = []
    for _, group in sorted_ratings.groupby("User_ID", sort=False):
        n_test = min(leave_last_n, len(group))
        test_idx = group.tail(n_test).index
        split_parts.append((group.drop(test_idx), group.loc[test_idx]))

    train_df = pd.concat([part[0] for part in split_parts], ignore_index=True)
    test_df = pd.concat([part[1] for part in split_parts], ignore_index=True)

    train_df = train_df[train_df["User_ID"].isin(train_df["User_ID"].value_counts().index)]
    test_df = test_df[test_df["User_ID"].isin(train_df["User_ID"].unique())]
    return train_df, test_df


def build_user_item_matrix(train_df: pd.DataFrame) -> pd.DataFrame:
    matrix = train_df.pivot_table(index="User_ID", columns="Movie_ID", values="Rating", fill_value=0)
    return matrix.sort_index()


def recommend_baseline(user_item: pd.DataFrame, sim_matrix: np.ndarray, user_id: int, k_neighbors: int, top_k: int) -> List[int]:
    """User-user CF recommendations following recommender.py's baseline behavior."""
    if user_id not in user_item.index:
        return []

    user_pos = user_item.index.get_loc(user_id)
    similarities = sim_matrix[user_pos].copy()
    similarities[user_pos] = -np.inf

    neighbor_count = min(k_neighbors, len(user_item.index) - 1)
    if neighbor_count <= 0:
        return []

    neighbor_positions = np.argpartition(similarities, -neighbor_count)[-neighbor_count:]
    neighbor_ids = user_item.index[neighbor_positions]

    scored_movies = user_item.loc[neighbor_ids].mean(axis=0).sort_values(ascending=False)
    seen_movies = set(user_item.columns[user_item.loc[user_id] > 0])
    recs = [int(movie) for movie in scored_movies.index if movie not in seen_movies]
    return recs[:top_k]


def recommend_popularity(train_df: pd.DataFrame, user_seen: Set[int], top_k: int) -> List[int]:
    popularity = (
        train_df.groupby("Movie_ID")["Rating"]
        .agg([("count", "count"), ("mean", "mean")])
        .sort_values(["count", "mean"], ascending=False)
    )
    recs = [int(mid) for mid in popularity.index if int(mid) not in user_seen]
    return recs[:top_k]


def recommend_random(catalog: Sequence[int], user_seen: Set[int], top_k: int, rng: np.random.Generator) -> List[int]:
    candidates = [mid for mid in catalog if mid not in user_seen]
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return list(candidates)
    sampled = rng.choice(candidates, size=top_k, replace=False)
    return [int(x) for x in sampled]


def precision_at_k(recommended: Sequence[int], relevant: Set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / min(k, len(rec_k))


def recall_at_k(recommended: Sequence[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: Sequence[int], relevant: Set[int], k: int) -> float:
    rec_k = recommended[:k]
    if not rec_k or not relevant:
        return 0.0

    dcg = 0.0
    for idx, item in enumerate(rec_k, start=1):
        rel = 1.0 if item in relevant else 0.0
        dcg += rel / np.log2(idx + 1)

    ideal_hits = min(len(relevant), len(rec_k))
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def catalog_coverage(recommendations: Dict[int, Sequence[int]], catalog: Iterable[int]) -> float:
    catalog_set = set(catalog)
    if not catalog_set:
        return 0.0
    recommended_items = {item for recs in recommendations.values() for item in recs}
    return len(recommended_items & catalog_set) / len(catalog_set)


def evaluate_variant(
    variant: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_item: pd.DataFrame,
    sim_matrix: np.ndarray,
    k: int,
    leave_last_n: int,
    neighbors: int,
    rng_seed: int,
) -> EvalResult:
    test_truth = test_df.groupby("User_ID")["Movie_ID"].apply(lambda x: set(map(int, x))).to_dict()
    train_seen = train_df.groupby("User_ID")["Movie_ID"].apply(lambda x: set(map(int, x))).to_dict()

    catalog = sorted(user_item.columns.astype(int).tolist())
    rng = np.random.default_rng(rng_seed)

    user_recs: Dict[int, List[int]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    ndcgs: List[float] = []

    for user_id, relevant in test_truth.items():
        seen = train_seen.get(user_id, set())

        if variant == "baseline":
            recs = recommend_baseline(user_item, sim_matrix, user_id, neighbors, k)
        elif variant == "popularity":
            recs = recommend_popularity(train_df, seen, k)
        elif variant == "random":
            recs = recommend_random(catalog, seen, k, rng)
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        user_recs[user_id] = recs
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))

    return EvalResult(
        variant=variant,
        precision_at_k=float(np.mean(precisions)) if precisions else 0.0,
        recall_at_k=float(np.mean(recalls)) if recalls else 0.0,
        ndcg_at_k=float(np.mean(ndcgs)) if ndcgs else 0.0,
        catalog_coverage=float(catalog_coverage(user_recs, catalog)),
        users_evaluated=len(test_truth),
        interactions_train=len(train_df),
        interactions_test=len(test_df),
        k=k,
        leave_last_n=leave_last_n,
    )


def append_results_csv(results_path: Path, dataset_path: str, results: Sequence[EvalResult]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp_utc",
        "dataset_path",
        "variant",
        "precision_at_k",
        "recall_at_k",
        "ndcg_at_k",
        "catalog_coverage",
        "users_evaluated",
        "interactions_train",
        "interactions_test",
        "k",
        "leave_last_n",
    ]

    file_exists = results_path.exists()
    now = datetime.now(timezone.utc).isoformat()

    with results_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "timestamp_utc": now,
                    "dataset_path": dataset_path,
                    "variant": result.variant,
                    "precision_at_k": f"{result.precision_at_k:.6f}",
                    "recall_at_k": f"{result.recall_at_k:.6f}",
                    "ndcg_at_k": f"{result.ndcg_at_k:.6f}",
                    "catalog_coverage": f"{result.catalog_coverage:.6f}",
                    "users_evaluated": result.users_evaluated,
                    "interactions_train": result.interactions_train,
                    "interactions_test": result.interactions_test,
                    "k": result.k,
                    "leave_last_n": result.leave_last_n,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation for collaborative filtering variants.")
    parser.add_argument("--dataset", default="Movie_data.csv", help="Path to ratings CSV.")
    parser.add_argument("--variant", default="all", choices=["baseline", "popularity", "random", "all"])
    parser.add_argument("--k", type=int, default=10, help="Top-K cutoff for ranking metrics.")
    parser.add_argument("--leave-last-n", type=int, default=1, help="Number of latest interactions per user in test.")
    parser.add_argument("--neighbors", type=int, default=10, help="Number of neighbors for baseline variant.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stochastic variants.")
    parser.add_argument(
        "--results-path",
        default="artifacts/offline_eval_results.csv",
        help="CSV file used to append evaluation history.",
    )
    args = parser.parse_args()

    ratings = load_ratings(args.dataset)
    train_df, test_df = leave_last_n_split(ratings, leave_last_n=args.leave_last_n)
    user_item = build_user_item_matrix(train_df)
    sim_matrix = cosine_similarity(user_item)

    variants = [args.variant] if args.variant != "all" else ["baseline", "popularity", "random"]

    results = [
        evaluate_variant(
            variant=variant,
            train_df=train_df,
            test_df=test_df,
            user_item=user_item,
            sim_matrix=sim_matrix,
            k=args.k,
            leave_last_n=args.leave_last_n,
            neighbors=args.neighbors,
            rng_seed=args.seed,
        )
        for variant in variants
    ]

    append_results_csv(Path(args.results_path), args.dataset, results)

    print("Offline evaluation complete")
    for result in results:
        print(
            f"[{result.variant}] "
            f"precision@{result.k}={result.precision_at_k:.4f}, "
            f"recall@{result.k}={result.recall_at_k:.4f}, "
            f"ndcg@{result.k}={result.ndcg_at_k:.4f}, "
            f"coverage={result.catalog_coverage:.4f}"
        )
    print(f"Results appended to: {args.results_path}")


if __name__ == "__main__":
    main()

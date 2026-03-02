import re
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def movie_recommender(user_item_m, X_user, user, k=10, top_n=10):
    """
    Provides movie recommendations based on collaborative filtering

    Parameters:
    - user_item_m: User-Item matrix (ratings dataframe)
    - X_user: User similarity matrix
    - user: User ID to get recommendations for
    - k: Number of similar users to consider
    - top_n: Number of top recommendations to return

    Returns:
    - DataFrame with recommended movie IDs
    """
    # Get the location of the actual user in the User-Items matrix
    # Use it to index the User similarity matrix
    user_similarities = X_user[user]
    # obtain the indices of the top k most similar users
    most_similar_users = user_item_m.index[user_similarities.argpartition(-k)[-k:]]
    # Obtain the mean ratings of those users for all movies
    rec_movies = user_item_m.loc[most_similar_users].mean(0).sort_values(ascending=False)
    # Discard already seen movies
    m_seen_movies = user_item_m.loc[user].gt(0)
    seen_movies = m_seen_movies.index[m_seen_movies].tolist()
    rec_movies = rec_movies.drop(seen_movies).head(top_n)
    # return recommendations - top similar users rated movies
    rec_movies_a = rec_movies.index.to_frame().reset_index(drop=True)
    rec_movies_a.rename(columns={rec_movies_a.columns[0]: "Movie_ID"}, inplace=True)
    return rec_movies_a


def prepare_data():
    """
    Load and prepare movie data

    Returns:
    - movies_df: Main movies dataframe
    - ratings_df: User-Item ratings matrix
    - rating_cosine_similarity: Cosine similarity matrix
    """
    # Load the rating data into a DataFrame:
    column_names = ["User_ID", "User_Names", "Movie_ID", "Rating", "Timestamp"]
    movies_df = pd.read_csv("Movie_data.csv", sep=",", names=column_names)

    # Load the movie information in a DataFrame:
    movies_title_df = pd.read_csv("Movie_Id_Titles.csv")
    movies_title_df.rename(columns={"item_id": "Movie_ID", "title": "Movie_Title"}, inplace=True)

    # Merge the DataFrames:
    movies_df = pd.merge(movies_df, movies_title_df, on="Movie_ID")

    # Create user-item matrix (ratings)
    ratings = movies_df.pivot_table(index="User_ID", columns="Movie_ID", values="Rating")
    ratings = ratings.fillna(0)
    ratings_df = pd.DataFrame(ratings)

    # Calculate cosine similarity
    rating_cosine_similarity = cosine_similarity(ratings)

    return movies_df, ratings_df, rating_cosine_similarity, movies_title_df


@lru_cache(maxsize=1)
def _get_default_data():
    return prepare_data()


def _normalize_to_unit(values):
    values = values.astype(float)
    vmin = values.min()
    vmax = values.max()
    if vmax == vmin:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - vmin) / (vmax - vmin)


def _get_user_seen_movies(ratings_df, user_id):
    seen_mask = ratings_df.loc[user_id].gt(0)
    return set(seen_mask.index[seen_mask].tolist())


def _extract_franchise_key(title):
    base = re.sub(r"\([^)]*\)", "", str(title)).strip().lower()
    base = re.sub(r"[^a-z0-9\s]", " ", base)
    base = re.sub(r"\s+", " ", base)
    if not base:
        return "unknown"
    tokens = base.split()
    # Truncate before a sequel-like suffix (2, ii, iii, etc.)
    cleaned = []
    for tok in tokens:
        if re.fullmatch(r"(\d+|i{1,4}|v|vi{0,3}|x)", tok):
            break
        cleaned.append(tok)
    return " ".join(cleaned[:3]) if cleaned else tokens[0]


def generate_candidates(
    user_id,
    ratings_df,
    rating_cosine_similarity,
    movies_df,
    candidate_size=100,
    k_similar=30,
):
    """Build a candidate set from CF, popularity, and trending sources."""
    if user_id not in ratings_df.index:
        return []

    seen_movies = _get_user_seen_movies(ratings_df, user_id)
    candidate_pool = []

    # 1) collaborative filtering source
    user_pos = ratings_df.index.get_loc(user_id)
    user_similarities = rating_cosine_similarity[user_pos]
    similar_positions = np.argpartition(user_similarities, -k_similar)[-k_similar:]
    similar_users = ratings_df.index[similar_positions]
    cf_scores = ratings_df.loc[similar_users].mean(0).sort_values(ascending=False)
    candidate_pool.extend([mid for mid in cf_scores.index.tolist() if mid not in seen_movies])

    # 2) popularity source
    popularity = (
        movies_df.groupby("Movie_ID")
        .agg(pop_count=("Rating", "count"), pop_mean=("Rating", "mean"))
        .assign(popularity=lambda x: x["pop_count"] * x["pop_mean"])
        .sort_values("popularity", ascending=False)
    )
    candidate_pool.extend([mid for mid in popularity.index.tolist() if mid not in seen_movies])

    # 3) recency/trending source
    trending = (
        movies_df.groupby("Movie_ID")
        .agg(recent_ts=("Timestamp", "max"), recent_mean=("Rating", "mean"))
        .assign(trending=lambda x: x["recent_ts"] * x["recent_mean"])
        .sort_values(["trending", "recent_mean"], ascending=False)
    )
    candidate_pool.extend([mid for mid in trending.index.tolist() if mid not in seen_movies])

    # deduplicate while preserving source order
    deduped = list(dict.fromkeys(candidate_pool))
    return deduped[:candidate_size]


def rank_candidates(
    user_id,
    candidate_ids,
    ratings_df,
    rating_cosine_similarity,
    movies_df,
):
    """Score candidates with a weighted blend of CF + popularity + recency + user affinity."""
    if not candidate_ids:
        return pd.DataFrame(columns=["Movie_ID", "final_score"])

    user_pos = ratings_df.index.get_loc(user_id)
    user_similarities = rating_cosine_similarity[user_pos]
    similar_users = ratings_df.index[np.argpartition(user_similarities, -30)[-30:]]

    candidate_index = pd.Index(candidate_ids, name="Movie_ID")

    cf_component = ratings_df.loc[similar_users, candidate_ids].mean(0)
    cf_component = _normalize_to_unit(cf_component)

    pop_stats = movies_df.groupby("Movie_ID").agg(pop_count=("Rating", "count"), pop_mean=("Rating", "mean"))
    pop_component = (pop_stats["pop_count"] * pop_stats["pop_mean"]).reindex(candidate_index).fillna(0)
    pop_component = _normalize_to_unit(pop_component)

    rec_stats = movies_df.groupby("Movie_ID").agg(latest_ts=("Timestamp", "max"))
    recency_component = rec_stats["latest_ts"].reindex(candidate_index).fillna(0)
    recency_component = _normalize_to_unit(recency_component)

    # user affinity: how close movie average is to user's mean rating tendency
    user_ratings = ratings_df.loc[user_id]
    rated = user_ratings[user_ratings > 0]
    user_mean = rated.mean() if not rated.empty else 3.0
    movie_mean = pop_stats["pop_mean"].reindex(candidate_index).fillna(user_mean)
    affinity_component = 1 - ((movie_mean - user_mean).abs() / 5.0)
    affinity_component = affinity_component.clip(lower=0, upper=1)

    scores = pd.DataFrame(
        {
            "Movie_ID": candidate_ids,
            "cf_score": cf_component.values,
            "popularity_score": pop_component.values,
            "recency_score": recency_component.values,
            "affinity_score": affinity_component.values,
        }
    )
    scores["final_score"] = (
        0.50 * scores["cf_score"]
        + 0.20 * scores["popularity_score"]
        + 0.15 * scores["recency_score"]
        + 0.15 * scores["affinity_score"]
    )
    return scores.sort_values("final_score", ascending=False).reset_index(drop=True)


def rerank_with_constraints(user_id, ranked_candidates, ratings_df, movies_df, top_n=10):
    """Apply seen-filtering, novelty boost, and simple title-based diversity constraints."""
    if ranked_candidates.empty:
        return ranked_candidates

    seen_movies = _get_user_seen_movies(ratings_df, user_id)
    popularity_count = movies_df.groupby("Movie_ID")["Rating"].count()
    popularity_norm = _normalize_to_unit(popularity_count)

    title_map = (
        movies_df[["Movie_ID", "Movie_Title"]]
        .drop_duplicates(subset=["Movie_ID"])
        .set_index("Movie_ID")["Movie_Title"]
    )

    candidates = ranked_candidates[~ranked_candidates["Movie_ID"].isin(seen_movies)].copy()
    if candidates.empty:
        return candidates

    selected_rows = []
    used_franchises = set()

    for _, row in candidates.sort_values("final_score", ascending=False).iterrows():
        movie_id = row["Movie_ID"]
        title = title_map.get(movie_id, "")
        franchise = _extract_franchise_key(title)
        novelty = 1 - popularity_norm.get(movie_id, 0)
        adjusted_score = row["final_score"] + 0.05 * novelty

        if franchise in used_franchises and len(selected_rows) < top_n:
            adjusted_score -= 0.10

        row_payload = row.copy()
        row_payload["final_score"] = adjusted_score
        row_payload["franchise"] = franchise
        selected_rows.append(row_payload)

        if franchise != "unknown":
            used_franchises.add(franchise)

        if len(selected_rows) >= max(top_n * 3, top_n):
            break

    reranked = pd.DataFrame(selected_rows).sort_values("final_score", ascending=False).head(top_n)
    return reranked.reset_index(drop=True)


def recommend(
    user_name,
    top_n=10,
    variant=None,
    movies_df=None,
    ratings_df=None,
    rating_cosine_similarity=None,
    movies_title_df=None,
):
    """High-level recommendation wrapper for the app and API callers."""
    if any(x is None for x in [movies_df, ratings_df, rating_cosine_similarity, movies_title_df]):
        movies_df, ratings_df, rating_cosine_similarity, movies_title_df = _get_default_data()

    user_rows = movies_df.loc[movies_df["User_Names"] == user_name, "User_ID"]
    if user_rows.empty:
        return pd.DataFrame(columns=["Movie_ID", "Movie_Title", "final_score"])

    user_id = user_rows.iloc[0]

    candidate_size = 100
    if variant == "fast":
        candidate_size = 60
    elif variant == "explore":
        candidate_size = 150

    candidate_ids = generate_candidates(
        user_id=user_id,
        ratings_df=ratings_df,
        rating_cosine_similarity=rating_cosine_similarity,
        movies_df=movies_df,
        candidate_size=candidate_size,
    )
    ranked = rank_candidates(
        user_id=user_id,
        candidate_ids=candidate_ids,
        ratings_df=ratings_df,
        rating_cosine_similarity=rating_cosine_similarity,
        movies_df=movies_df,
    )
    reranked = rerank_with_constraints(
        user_id=user_id,
        ranked_candidates=ranked,
        ratings_df=ratings_df,
        movies_df=movies_df,
        top_n=top_n,
    )

    result = reranked[["Movie_ID", "final_score"]].merge(movies_title_df, how="left", on="Movie_ID")
    return result[["Movie_ID", "Movie_Title", "final_score"]]


def movie_recommender_run(user_Name, movies_df, ratings_df, rating_cosine_similarity, movies_title_df):
    """Backward-compatible wrapper that now uses the modular recommendation pipeline."""
    return recommend(
        user_name=user_Name,
        top_n=10,
        movies_df=movies_df,
        ratings_df=ratings_df,
        rating_cosine_similarity=rating_cosine_similarity,
        movies_title_df=movies_title_df,
    )

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def movie_recommender(user_item_m, X_user, user, k=10, top_n=50):
    """Return collaborative-filtering candidate movie IDs for a user."""
    user_similarities = X_user[user]
    k = min(k, len(user_item_m.index))
    most_similar_users = user_item_m.index[user_similarities.argpartition(-k)[-k:]]
    rec_movies = user_item_m.loc[most_similar_users].mean(0).sort_values(ascending=False)

    seen_mask = user_item_m.loc[user].gt(0)
    seen_movies = seen_mask.index[seen_mask].tolist()
    rec_movies = rec_movies.drop(seen_movies, errors="ignore").head(top_n)

    rec_movies_a = rec_movies.index.to_frame().reset_index(drop=True)
    rec_movies_a.rename(columns={rec_movies_a.columns[0]: "Movie_ID"}, inplace=True)
    rec_movies_a["cf_score"] = rec_movies.values
    return rec_movies_a


def prepare_data():
    """Load and prepare movie data and similarity matrix."""
    column_names = ["User_ID", "User_Names", "Movie_ID", "Rating", "Timestamp"]
    movies_df = pd.read_csv("Movie_data.csv", sep=",", names=column_names)

    movies_title_df = pd.read_csv("Movie_Id_Titles.csv")
    movies_title_df.rename(columns={"item_id": "Movie_ID", "title": "Movie_Title"}, inplace=True)

    movies_df = pd.merge(movies_df, movies_title_df, on="Movie_ID")

    ratings = movies_df.pivot_table(index="User_ID", columns="Movie_ID", values="Rating")
    ratings = ratings.fillna(0)
    ratings_df = pd.DataFrame(ratings)

    rating_cosine_similarity = cosine_similarity(ratings)

    # popularity prior for ranking
    popularity_df = (
        movies_df.groupby("Movie_ID")["Rating"]
        .agg(["count", "mean"])
        .rename(columns={"count": "rating_count", "mean": "rating_mean"})
        .reset_index()
    )
    popularity_df["popularity_score"] = (
        popularity_df["rating_count"] * popularity_df["rating_mean"]
    )

    return movies_df, ratings_df, rating_cosine_similarity, movies_title_df, popularity_df


def generate_candidates(user_id, ratings_df, rating_cosine_similarity, popularity_df, n_candidates=100):
    """Generate candidates from collaborative filtering and popularity fallback."""
    cf_candidates = movie_recommender(
        ratings_df,
        rating_cosine_similarity,
        user_id,
        k=10,
        top_n=n_candidates,
    )

    seen_mask = ratings_df.loc[user_id].gt(0)
    seen_movies = set(seen_mask.index[seen_mask].tolist())

    popular_fallback = popularity_df[~popularity_df["Movie_ID"].isin(seen_movies)].nlargest(
        n_candidates, "popularity_score"
    )[["Movie_ID", "popularity_score"]]

    candidates = cf_candidates.merge(popular_fallback, how="outer", on="Movie_ID")
    return candidates.fillna(0)


def rank_candidates(user_id, candidates, ratings_df, popularity_df):
    """Rank candidates with weighted score blending CF and popularity."""
    user_mean_rating = ratings_df.loc[user_id][ratings_df.loc[user_id] > 0].mean()
    user_mean_rating = user_mean_rating if pd.notna(user_mean_rating) else 3.0

    pop_map = popularity_df.set_index("Movie_ID")["rating_mean"]
    candidates["rating_mean"] = candidates["Movie_ID"].map(pop_map).fillna(user_mean_rating)

    candidates["rank_score"] = (
        0.65 * candidates.get("cf_score", 0)
        + 0.20 * (candidates.get("popularity_score", 0) / (candidates.get("popularity_score", 0).max() + 1e-9))
        + 0.15 * (candidates["rating_mean"] / 5.0)
    )
    return candidates.sort_values("rank_score", ascending=False)


def recommend(user_name, movies_df, ratings_df, rating_cosine_similarity, movies_title_df, popularity_df, top_n=10, variant="control"):
    """Return ranked recommendations for a user and variant."""
    user_id = movies_df.loc[movies_df["User_Names"] == user_name].User_ID.values[0]

    if variant == "control":
        temp = movie_recommender(ratings_df, rating_cosine_similarity, user_id, top_n=top_n)
        top_k_rec = temp.merge(movies_title_df, how="inner")
        top_k_rec["rank_score"] = top_k_rec.get("cf_score", 0)
        return user_id, top_k_rec.head(top_n)

    candidates = generate_candidates(user_id, ratings_df, rating_cosine_similarity, popularity_df)
    ranked = rank_candidates(user_id, candidates, ratings_df, popularity_df).head(top_n)
    top_k_rec = ranked.merge(movies_title_df, how="inner", on="Movie_ID")
    return user_id, top_k_rec.head(top_n)

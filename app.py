import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from analytics import summarize_experiment
from experiments import EXPERIMENT_NAME, assign_variant
from recommender import prepare_data, recommend


st.set_page_config(layout="wide", page_title="Movie Recommendation App", page_icon=":Cinema:")


@st.cache_resource
def load_recommendation_data():
    return prepare_data()


movies_df, ratings_df, rating_cosine_similarity, movies_title_df, popularity_df = load_recommendation_data()


LOG_PATH = Path("logs/events.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_event(event_type, user_id, variant, movie_id=None, rank_position=None):
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_NAME,
        "event_type": event_type,
        "user_id": int(user_id),
        "variant": variant,
        "movie_id": int(movie_id) if movie_id is not None else None,
        "rank_position": int(rank_position) if rank_position is not None else None,
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


n_users = movies_df.User_Names.unique()
st.header("Collaborative Filtering Recommendation System")

user_name = st.selectbox("Select a user name:", (n_users))

user_id = movies_df.loc[movies_df["User_Names"] == user_name].User_ID.values[0]
variant = assign_variant(user_id)
st.caption(f"Experiment: {EXPERIMENT_NAME} | Assigned variant: {variant}")

_, result = recommend(
    user_name,
    movies_df,
    ratings_df,
    rating_cosine_similarity,
    movies_title_df,
    popularity_df,
    top_n=10,
    variant=variant,
)

st.write("This user might be interested in the following movies:")
st.table(result[["Movie_Title"]])

if st.button("Log impressions"):
    for pos, row in result.reset_index(drop=True).iterrows():
        log_event("impression", user_id, variant, row["Movie_ID"], pos + 1)
    st.success("Impressions logged.")

st.subheader("Feedback (for A/B testing)")
for pos, row in result.reset_index(drop=True).iterrows():
    c1, c2, c3 = st.columns([4, 1, 1])
    c1.write(f"{pos + 1}. {row['Movie_Title']}")
    if c2.button("Clicked", key=f"click_{row['Movie_ID']}"):
        log_event("click", user_id, variant, row["Movie_ID"], pos + 1)
        st.toast(f"Logged click for {row['Movie_Title']}")
    if c3.button("Liked", key=f"like_{row['Movie_ID']}"):
        log_event("like", user_id, variant, row["Movie_ID"], pos + 1)
        st.toast(f"Logged like for {row['Movie_Title']}")

ids = result.Movie_ID
names = result.Movie_Title
fig = make_subplots(
    rows=5,
    cols=2,
    subplot_titles=(names),
    specs=[
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "bar"}, {"type": "bar"}],
    ],
)

for i in range(len(result)):
    temp = (
        movies_df.loc[movies_df["Movie_ID"] == ids.iloc[i]]
        .groupby("Rating")
        .User_ID.count()
        .reset_index()
    )

    user_counts = temp.User_ID.to_numpy()

    x_row = int(i / 2 + 1)
    y_col = i % 2 + 1

    fig.add_trace(go.Bar(x=[1, 2, 3, 4, 5], y=user_counts), row=x_row, col=y_col)
    fig.update_xaxes(title_text="Rating", row=x_row, col=y_col)
    fig.update_yaxes(title_text="Users", row=x_row, col=y_col)

fig.update_layout(height=900, width=800, showlegend=False, title="Ratings of Suggested Movies")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Experiment analytics")
summary_df = summarize_experiment()
if summary_df.empty:
    st.info("No events logged yet.")
else:
    show_df = summary_df.copy()
    show_df[["ctr", "like_rate"]] = show_df[["ctr", "like_rate"]].round(4)
    st.dataframe(show_df)

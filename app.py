import pandas as pd
import numpy as np
import streamlit as st
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from recommender import prepare_data, movie_recommender_run


EVENT_LOG_PATH = Path("logs/events.jsonl")
EXPERIMENT_VARIANT = "baseline_v1"


def _get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    return st.session_state.session_id


def log_event(event_type, user_id, variant, rank_position, movie_id, movie_title=None, metadata=None):
    event_record = {
        "schema_version": 1,
        "event_id": str(uuid4()),
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": str(user_id),
        "session_id": _get_session_id(),
        "variant": variant,
        "rank_position": int(rank_position),
        "movie_id": int(movie_id),
        "movie_title": movie_title,
        "metadata": metadata or {},
    }

    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG_PATH.open("a", encoding="utf-8") as event_file:
        event_file.write(json.dumps(event_record) + "\n")

#Set page configuration
st.set_page_config(layout = "wide", page_title = "Movie Recommendation App", page_icon = ":Cinema:")

# Load data once and cache it
@st.cache_resource
def load_recommendation_data():
    return prepare_data()

movies_df, ratings_df, rating_cosine_similarity, movies_title_df = load_recommendation_data()

#Display movie rating charts here
#Read the dataset to find unique users
n_users = movies_df.User_Names.unique()

#Create application's header
st.header("Collaborative Filtering Recommendation System")

#Create a dropdown of UserIDs
User_Name = st.selectbox(
 "Select a user name:",
 (n_users)
)

_get_session_id()

st.write("This user might be interested in the following movies:")
#Find and display recommendations for selected users
result = movie_recommender_run(User_Name, movies_df, ratings_df, rating_cosine_similarity, movies_title_df)
st.table(result.Movie_Title)

current_signature = (
    str(User_Name),
    EXPERIMENT_VARIANT,
    tuple(int(movie_id) for movie_id in result.Movie_ID.tolist()),
)

if st.session_state.get("last_impression_signature") != current_signature:
    for rank_position, (_, recommended_movie) in enumerate(result.iterrows(), start=1):
        log_event(
            event_type="impression",
            user_id=User_Name,
            variant=EXPERIMENT_VARIANT,
            rank_position=rank_position,
            movie_id=recommended_movie["Movie_ID"],
            movie_title=recommended_movie["Movie_Title"],
        )
    st.session_state.last_impression_signature = current_signature

st.subheader("Recommendation interactions")
for rank_position, (_, recommended_movie) in enumerate(result.iterrows(), start=1):
    movie_id = int(recommended_movie["Movie_ID"])
    movie_title = recommended_movie["Movie_Title"]
    st.write(f"{rank_position}. {movie_title}")

    interaction_columns = st.columns(3)

    if interaction_columns[0].button("Click", key=f"click_{movie_id}_{rank_position}"):
        log_event(
            event_type="click",
            user_id=User_Name,
            variant=EXPERIMENT_VARIANT,
            rank_position=rank_position,
            movie_id=movie_id,
            movie_title=movie_title,
        )

    if interaction_columns[1].button("Like", key=f"like_{movie_id}_{rank_position}"):
        log_event(
            event_type="like",
            user_id=User_Name,
            variant=EXPERIMENT_VARIANT,
            rank_position=rank_position,
            movie_id=movie_id,
            movie_title=movie_title,
        )

    if interaction_columns[2].button("Watch Start", key=f"watch_start_{movie_id}_{rank_position}"):
        log_event(
            event_type="watch_start",
            user_id=User_Name,
            variant=EXPERIMENT_VARIANT,
            rank_position=rank_position,
            movie_id=movie_id,
            movie_title=movie_title,
        )

# Display details of provided recommendations
ids= result.Movie_ID
Names=result.Movie_Title
fig = make_subplots(
    rows=5, cols=2,
    subplot_titles=(Names),
    specs=[
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "bar"}, {"type": "bar"}] 
    ])

# x_row and y_col will determine the location of a plot in the plot-grid
x_row=1
y_col=1
for i in range (len(result)):
    temp=(movies_df.loc[movies_df['Movie_ID'] == ids[i]]).groupby('Rating').User_ID.count().reset_index()
    
    Rating=temp.Rating.to_numpy()
    User_ID= temp.User_ID.to_numpy()

    x_row= int( i/2 +1)
    y_col= i%2 + 1
    
    fig.add_trace(go.Bar(x=[1,2,3,4,5], y=User_ID), row=x_row, col=y_col)
    fig.update_xaxes(title_text="Rating", row=x_row, col=y_col)
    fig.update_yaxes(title_text="Users", row=x_row, col=y_col)

fig.update_layout(height=900,width=800, showlegend=False, title= "Ratings of Suggested Movies")

st.plotly_chart(fig, use_container_width=True)

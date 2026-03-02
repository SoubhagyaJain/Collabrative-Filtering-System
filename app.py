import logging
import pandas as pd
import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from recommender import prepare_data, movie_recommender_run
from experiments import (
    EXPERIMENT_NAME,
    EXPERIMENT_START_DATE,
    EXPERIMENT_TRAFFIC_SPLIT,
    EXPERIMENT_SUCCESS_METRICS,
    assign_variant,
)

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

st.write("This user might be interested in the following movies:")

user_id = movies_df.loc[movies_df["User_Names"] == User_Name].User_ID.values[0]
variant = assign_variant(user_id, EXPERIMENT_NAME, EXPERIMENT_TRAFFIC_SPLIT)

# Find and display recommendations for selected users
result = movie_recommender_run(
    User_Name,
    movies_df,
    ratings_df,
    rating_cosine_similarity,
    movies_title_df,
    variant=variant,
)

logging.info(
    "recommendation_request experiment=%s start_date=%s user_id=%s variant=%s metrics=%s",
    EXPERIMENT_NAME,
    EXPERIMENT_START_DATE,
    user_id,
    variant,
    ",".join(EXPERIMENT_SUCCESS_METRICS),
)

st.caption(f"Experiment variant: {variant}")
st.table(result.Movie_Title)

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


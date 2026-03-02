import pandas as pd
import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from recommender import prepare_data, movie_recommender_run
from analytics import load_event_logs, compute_variant_metrics

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
#Find and display recommendations for selected users
result = movie_recommender_run(User_Name, movies_df, ratings_df, rating_cosine_similarity, movies_title_df)
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



# Admin/debug analytics panel for A/B event logs
with st.expander("Admin / Debug: Experiment analytics", expanded=False):
    st.caption("Compute per-variant metrics from event logs (CSV).")
    logs_path = st.text_input("Event log CSV path", value="event_logs.csv")
    baseline_variant = st.text_input("Baseline variant (optional)", value="")

    try:
        logs_df = load_event_logs(logs_path)
        if logs_df.empty:
            st.info("No logs found (or file is empty). Add a CSV with `variant` and `event_type` columns.")
        else:
            metrics_df = compute_variant_metrics(logs_df, baseline_variant=baseline_variant or None)
            formatted = metrics_df.copy()
            pct_cols = [
                "ctr",
                "ctr_ci_low",
                "ctr_ci_high",
                "engagement_rate",
                "engagement_ci_low",
                "engagement_ci_high",
                "ctr_uplift_vs_baseline",
            ]
            for col in pct_cols:
                if col in formatted.columns:
                    formatted[col] = formatted[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "-")

            if "avg_rank_clicked" in formatted.columns:
                formatted["avg_rank_clicked"] = formatted["avg_rank_clicked"].map(
                    lambda x: f"{x:.2f}" if pd.notna(x) else "-"
                )

            st.dataframe(formatted, use_container_width=True)
            st.caption(
                "Includes sample sizes (`impressions`) and Wilson 95% confidence intervals for CTR and engagement rate."
            )
    except Exception as exc:
        st.error(f"Unable to compute analytics: {exc}")

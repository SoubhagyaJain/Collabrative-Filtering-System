import pandas as pd
import numpy as np
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
    rec_movies_a.rename(columns={rec_movies_a.columns[0]: 'Movie_ID'}, inplace=True)
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
    column_names = ['User_ID', 'User_Names', 'Movie_ID', 'Rating', 'Timestamp']
    movies_df = pd.read_csv('Movie_data.csv', sep=',', names=column_names)
    
    # Load the movie information in a DataFrame:
    movies_title_df = pd.read_csv("Movie_Id_Titles.csv")
    movies_title_df.rename(columns={'item_id': 'Movie_ID', 'title': 'Movie_Title'}, inplace=True)
    
    # Merge the DataFrames:
    movies_df = pd.merge(movies_df, movies_title_df, on='Movie_ID')
    
    # Create user-item matrix (ratings)
    ratings = movies_df.pivot_table(index='User_ID', columns='Movie_ID', values='Rating')
    ratings = ratings.fillna(0)
    ratings_df = pd.DataFrame(ratings)
    
    # Calculate cosine similarity
    rating_cosine_similarity = cosine_similarity(ratings)
    
    return movies_df, ratings_df, rating_cosine_similarity, movies_title_df


def movie_recommender_run(user_Name, movies_df, ratings_df, rating_cosine_similarity, movies_title_df, variant="control"):
    """
    Get recommendations for a specific user by name
    
    Parameters:
    - user_Name: Name of the user
    - movies_df: Movies dataframe
    - ratings_df: Ratings dataframe
    - rating_cosine_similarity: Similarity matrix
    - movies_title_df: Movie titles dataframe
    
    Returns:
    - DataFrame with recommended movies
    """
    # Get ID from Name
    user_ID = movies_df.loc[movies_df['User_Names'] == user_Name].User_ID.values[0]
    # Route recommendation strategy by experiment variant
    if variant == "treatment":
        # Placeholder: treatment pipeline can diverge here from CF baseline.
        temp = movie_recommender(ratings_df, rating_cosine_similarity, user_ID)
    else:
        temp = movie_recommender(ratings_df, rating_cosine_similarity, user_ID)
    # Join with the movie_title_df to get the movie titles
    top_k_rec = temp.merge(movies_title_df, how='inner')
    return top_k_rec

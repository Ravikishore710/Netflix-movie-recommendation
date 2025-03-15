import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 Load dataset
file_path = "C:/Users/Asus/Downloads/nnetflixData.csv/netflixData.csv"
df = pd.read_csv(file_path)

# 🔹 Keep relevant columns and remove missing values
df = df[['Title', 'Genres', 'Imdb Score']].dropna()

# 🔹 Convert IMDb Score to numeric format
df['Imdb Score'] = df['Imdb Score'].astype(str).str.extract(r'(\d+\.\d+)').astype(float)

# 🔹 Visualization 1: IMDb Score Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Imdb Score'], bins=20, kde=True, color='blue')
plt.xlabel('IMDb Score')
plt.ylabel('Count')
plt.title('Distribution of IMDb Scores')
plt.show()

# 🔹 One-hot encode genres for similarity calculations
encoder = OneHotEncoder()
genre_matrix = encoder.fit_transform(df[['Genres']])  # Convert genres to binary matrix

# 🔹 Create a simulated user-movie rating matrix using IMDb scores
num_users = 10  # Adjust the number of users
num_movies = len(df)

np.random.seed(42)
user_movie_matrix = np.zeros((num_users, num_movies))

for i in range(num_users):
    sampled_indices = np.random.choice(num_movies, size=num_movies//2, replace=False)  # Each user rates ~50% of movies
    user_movie_matrix[i, sampled_indices] = df.iloc[sampled_indices]['Imdb Score'].values

# 🔹 Apply SVD (Matrix Factorization)
num_latent_factors = min(20, user_movie_matrix.shape[0] - 1, user_movie_matrix.shape[1] - 1)

U, sigma, Vt = svds(user_movie_matrix, k=num_latent_factors)
sigma = np.diag(sigma)

# 🔹 Visualization 2: Singular Values (Latent Factor Importance)
plt.figure(figsize=(8,5))
plt.plot(sigma[::-1], marker='o', linestyle='--', color='red')
plt.xlabel("Latent Factor Index")
plt.ylabel("Singular Value")
plt.title("Importance of Latent Factors (Singular Values)")
plt.show()

# 🔹 Reconstruct the missing ratings
predicted_ratings = np.dot(np.dot(U, sigma), Vt)

# 🔹 Visualization 3: Actual vs. Predicted Ratings
plt.figure(figsize=(8,5))
plt.scatter(user_movie_matrix.flatten(), predicted_ratings.flatten(), alpha=0.5, color='green')
plt.xlabel("Actual IMDb Score")
plt.ylabel("Predicted IMDb Score")
plt.title("Actual vs Predicted IMDb Scores")
plt.show()

# 🔹 Compute similarity between movies based on genres
movie_similarity = cosine_similarity(genre_matrix)

# 🔹 Improved Recommendation Function
def recommend_movies(user_id=0, num_recommendations=5):
    """Recommend top movies for a user, ensuring already watched movies are excluded and providing explanations."""
    
    user_ratings = predicted_ratings[user_id]  # Get predicted ratings for the user
    watched_movies = np.where(user_movie_matrix[user_id] > 0)[0]  # Get indices of watched movies

    # Exclude watched movies from recommendations
    unwatched_indices = [i for i in range(len(user_ratings)) if i not in watched_movies]

    # Sort only unwatched movies based on predicted ratings (descending order)
    sorted_movie_indices = sorted(unwatched_indices, key=lambda x: user_ratings[x], reverse=True)

    # Get top recommended movie titles
    recommended_movie_indices = sorted_movie_indices[:num_recommendations]
    recommended_movies = df.iloc[recommended_movie_indices]['Title'].values

    # Find the most similar watched movie for each recommendation
    recommendations_with_reasons = []
    for rec_index in recommended_movie_indices:
        # Find the most similar watched movie based on genre similarity
        similar_movie_index = max(watched_movies, key=lambda x: movie_similarity[rec_index, x])
        similar_movie = df.iloc[similar_movie_index]['Title']
        recommendations_with_reasons.append((df.iloc[rec_index]['Title'], similar_movie))

    return recommendations_with_reasons

# 🔹 Get recommendations for a specific user
user_id = 9  # Change this to test different users
num_recommendations = 5
top_movies_with_reasons = recommend_movies(user_id=user_id, num_recommendations=num_recommendations)

# 🔹 Visualization 4: Top Recommended Movies for the User
top_movie_titles = [movie for movie, _ in top_movies_with_reasons]
plt.figure(figsize=(8,5))
sns.barplot(x=top_movie_titles, y=range(len(top_movie_titles)), palette="viridis")
plt.xlabel("Movie Title")
plt.ylabel("Ranking")
plt.title(f"Top {num_recommendations} Recommended Movies for User {user_id}")
plt.xticks(rotation=45)
plt.show()

# 🔹 Print recommendations in the correct format
print(f"\n🎬 **Top {num_recommendations} Recommended Movies for User {user_id}:**")
for i, (movie, reason) in enumerate(top_movies_with_reasons, 1):
    print(f"{i}. {movie} (Recommended because you watched: {reason})")

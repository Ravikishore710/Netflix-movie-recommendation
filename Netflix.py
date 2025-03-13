import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse.linalg import svds

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

# 🔹 Create a simulated user-movie rating matrix using IMDb scores
# Assuming each user has rated different movies
num_users = 10  # Adjust the number of users
num_movies = len(df)

# Randomly assign ratings from IMDb scores to different users
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

# 🔹 Function to recommend top movies for a user
def recommend_movies(user_id=0, num_recommendations=5):
    user_ratings = predicted_ratings[user_id]
    sorted_movie_indices = np.argsort(user_ratings)[::-1]  # Sort in descending order
    recommended_movie_titles = df.iloc[sorted_movie_indices[:num_recommendations]]['Title'].values
    return recommended_movie_titles

# 🔹 Get recommendations for a specific user
user_id = 0
top_movies = recommend_movies(user_id=user_id, num_recommendations=5)

# 🔹 Visualization 4: Top Recommended Movies for the User
plt.figure(figsize=(8,5))
sns.barplot(x=top_movies, y=range(len(top_movies)), palette="viridis")
plt.xlabel("Movie Title")
plt.ylabel("Ranking")
plt.title(f"Top {len(top_movies)} Recommended Movies for User {user_id}")
plt.xticks(rotation=45)
plt.show()

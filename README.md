# 🎬 Netflix Movie Recommendation  System

This project builds a movie recommendation system using **Collaborative Filtering** and **Matrix Factorization (SVD - Singular Value Decomposition)**. The dataset includes movie titles, genres, and IMDb scores, which are used to predict missing ratings and generate personalized movie recommendations for users.

The primary goal is to predict movie ratings for users and recommend movies they might like based on patterns in user preferences.

---

## 🔹 Step-by-Step Explanation of the Code

### 1️⃣ **Loading and Preprocessing the Data**
- The dataset is read using Pandas (`pd.read_csv`), and only relevant columns (`Title`, `Genres`, `IMDb Score`) are selected.
- Missing values are dropped to avoid errors in the recommendation process.
- IMDb scores are extracted as numeric values and converted into a float format.

### 2️⃣ **Visualizing IMDb Score Distribution**
- A histogram is plotted using `seaborn.histplot()` to display the frequency of different IMDb scores.
- This helps in understanding whether scores are normally distributed or skewed, which can influence the recommendation algorithm.

### 3️⃣ **Creating a User-Movie Rating Matrix**
- A user-movie rating matrix is created, where each row represents a user, each column represents a movie, and the values represent ratings.
- IMDb scores are used to simulate user ratings by randomly assigning movies to users using NumPy.
- A zero matrix is first created, and selected IMDb scores are assigned to random users.

### 4️⃣ **Performing SVD (Singular Value Decomposition)**
- SVD decomposes the user-movie rating matrix into three smaller matrices:
  - **U (User Preferences Matrix)**: Captures user-specific preferences.
  - **Σ (Sigma - Latent Factors)**: Represents the importance of different latent factors.
  - **Vt (Movie Features Matrix)**: Represents movie-related features.
- The Sigma matrix is diagonalized to reconstruct the original matrix.

### 5️⃣ **Visualizing Singular Values (Latent Factor Importance)**
- A line plot of singular values is generated to visualize the importance of latent factors.
- Singular values represent how much information each factor contributes to the model.

### 6️⃣ **Reconstructing the User-Movie Rating Matrix**
- The user-movie rating matrix is reconstructed by multiplying the decomposed matrices back together.
- This step fills in missing ratings, allowing us to predict how much a user might like a movie they haven't rated yet.

### 7️⃣ **Comparing Actual vs. Predicted Ratings**
- A scatter plot is used to compare actual IMDb scores with predicted IMDb scores from the SVD model.
- This helps evaluate the accuracy of predictions.

### 8️⃣ **Recommending Movies for a User**
- Predicted ratings for a specific user are extracted.
- Movies are sorted based on predicted ratings (in descending order).
- The top N movies are selected as recommendations.

### 9️⃣ **Visualizing Top Recommended Movies**
- The final visualization displays the top recommended movies for a specific user using a bar plot.
- Movies are plotted on the x-axis, and their ranking is on the y-axis.

---

## 🔹 **Why Use SVD for Movie Recommendation?**

### Advantages of SVD for Recommendation Systems:
- ✅ **Captures Hidden Relationships**: Finds latent features that connect users and movies.
- ✅ **Handles Sparse Data**: Works well with incomplete user-movie rating matrices.
- ✅ **Scalable**: Can be applied to large datasets efficiently.
- ✅ **Predicts Missing Ratings**: Helps recommend movies to users based on inferred preferences.

### Why Not Use Simple Filtering Techniques?
- Content-Based Filtering (based on genres, cast, etc.) ignores user preferences.
- Basic Collaborative Filtering (without Matrix Factorization) performs poorly on sparse datasets.
- SVD learns deeper patterns and reconstructs missing ratings more accurately.

---

## 🔹 **Summary**

This project successfully builds a **Netflix Movie Recommendation System** using SVD-based Collaborative Filtering.

### Key Takeaways:
- 🔹 IMDb Scores are used as real user ratings instead of random values.
- 🔹 Matrix Factorization (SVD) helps predict missing ratings.
- 🔹 Multiple visualizations are used to analyze the data and evaluate the model.
- 🔹 A user-movie rating matrix is created and optimized for recommendations.
- 🔹 The final output is a list of top recommended movies for a user.

This approach can be further improved using **Alternating Least Squares (ALS)**, **Deep Learning**, or **Hybrid Recommendation Models** for even better performance! 🚀

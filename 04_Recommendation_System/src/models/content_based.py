"""
Content-Based Filtering
Uses item features (genres, year, etc.) for recommendations
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.config import CB_FEATURE_WEIGHTS


class ContentBasedFiltering:
    """Content-based recommendation using item features"""
    
    def __init__(
        self,
        feature_weights: Optional[Dict[str, float]] = None
    ):
        self.feature_weights = feature_weights or CB_FEATURE_WEIGHTS
        self.item_features = None
        self.item_similarity_matrix = None
        self.user_item_matrix = None
        self.movies_df = None
        
    def fit(
        self,
        user_item_matrix: pd.DataFrame,
        movies_df: pd.DataFrame
    ):
        """Fit content-based model"""
        self.user_item_matrix = user_item_matrix.copy()
        self.movies_df = movies_df.copy()
        
        # Build item feature matrix
        self.item_features = self._build_feature_matrix(movies_df)
        
        # Compute item-item similarity
        self.item_similarity_matrix = pd.DataFrame(
            cosine_similarity(self.item_features),
            index=self.item_features.index,
            columns=self.item_features.index
        )
        
        return self
    
    def _build_feature_matrix(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix from movie metadata"""
        features_list = []
        item_ids = []
        
        for _, movie in movies_df.iterrows():
            item_id = movie['movie_id']
            item_ids.append(item_id)
            
            feature_vector = []
            
            # Genre features (one-hot encoded)
            if 'genres' in movie and isinstance(movie['genres'], list):
                genre_cols = [
                    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
                ]
                for genre in genre_cols:
                    feature_vector.append(1 if genre in movie['genres'] else 0)
            else:
                # Fallback: use genre columns directly
                genre_cols = [
                    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
                ]
                for genre in genre_cols:
                    if genre in movie:
                        feature_vector.append(float(movie[genre]))
                    else:
                        feature_vector.append(0.0)
            
            # Year feature (normalized)
            if 'year' in movie and pd.notna(movie['year']):
                # Normalize year to [0, 1] range (assuming years 1900-2020)
                year_norm = (movie['year'] - 1900) / 120.0
                feature_vector.append(year_norm)
            else:
                feature_vector.append(0.5)  # Default middle value
            
            features_list.append(feature_vector)
        
        # Create feature matrix
        feature_matrix = pd.DataFrame(
            features_list,
            index=item_ids
        )
        
        return feature_matrix
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating based on user's past preferences"""
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix.mean().mean()
        
        if item_id not in self.item_similarity_matrix.index:
            return self.user_item_matrix.loc[user_id].mean()
        
        # Get user's rated items
        user_ratings = self.user_item_matrix.loc[user_id]
        user_ratings = user_ratings[user_ratings > 0]
        
        if len(user_ratings) == 0:
            return self.user_item_matrix[item_id].mean()
        
        # Get similarity scores with user's rated items
        similarities = self.item_similarity_matrix[item_id]
        similarities = similarities[similarities.index.isin(user_ratings.index)]
        
        if len(similarities) == 0:
            return self.user_item_matrix[item_id].mean()
        
        # Weighted average: similarity * user_rating
        numerator = 0
        denominator = 0
        
        for rated_item_id, rating in user_ratings.items():
            if rated_item_id in similarities.index:
                similarity = similarities[rated_item_id]
                numerator += similarity * rating
                denominator += abs(similarity)
        
        if denominator == 0:
            return self.user_item_matrix[item_id].mean()
        
        prediction = numerator / denominator
        prediction = np.clip(prediction, 1, 5)
        
        return float(prediction)
    
    def recommend(
        self,
        user_id: int,
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[int, float]]:
        """Recommend items based on content similarity"""
        if user_id not in self.user_item_matrix.index:
            # Cold start: return popular items
            item_means = self.user_item_matrix.mean().sort_values(ascending=False)
            recommendations = [
                (item_id, rating) for item_id, rating in item_means.head(n_recommendations).items()
            ]
            return recommendations
        
        # Get user's rated items
        user_ratings = self.user_item_matrix.loc[user_id]
        user_ratings = user_ratings[user_ratings > 0]
        
        if len(user_ratings) == 0:
            # No ratings: return popular items
            item_means = self.user_item_matrix.mean().sort_values(ascending=False)
            recommendations = [
                (item_id, rating) for item_id, rating in item_means.head(n_recommendations).items()
            ]
            return recommendations
        
        # Get items user hasn't rated
        if exclude_rated:
            unrated_items = self.user_item_matrix.columns[
                self.user_item_matrix.loc[user_id] == 0
            ].tolist()
        else:
            unrated_items = self.user_item_matrix.columns.tolist()
        
        # Calculate scores for unrated items
        predictions = []
        for item_id in unrated_items:
            if item_id in self.item_similarity_matrix.index:
                pred_rating = self.predict(user_id, item_id)
                predictions.append((item_id, pred_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]


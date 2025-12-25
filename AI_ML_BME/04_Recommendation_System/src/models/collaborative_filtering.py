"""
Collaborative Filtering Recommendation Models
User-based and Item-based Collaborative Filtering
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy.sparse import csr_matrix

from src.config import CF_SIMILARITY_METRIC, CF_NEIGHBORS


class UserBasedCF:
    """User-based Collaborative Filtering"""
    
    def __init__(
        self,
        similarity_metric: str = CF_SIMILARITY_METRIC,
        n_neighbors: int = CF_NEIGHBORS
    ):
        self.similarity_metric = similarity_metric
        self.n_neighbors = n_neighbors
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_mean_ratings = None
        
    def fit(self, user_item_matrix: pd.DataFrame):
        """Fit the model on user-item matrix"""
        self.user_item_matrix = user_item_matrix.copy()
        
        # Calculate user mean ratings for centering
        self.user_mean_ratings = self.user_item_matrix.mean(axis=1)
        
        # Center the matrix (subtract user mean)
        centered_matrix = self.user_item_matrix.sub(self.user_mean_ratings, axis=0)
        centered_matrix = centered_matrix.fillna(0)
        
        # Compute similarity matrix
        if self.similarity_metric == "cosine":
            self.similarity_matrix = pd.DataFrame(
                cosine_similarity(centered_matrix),
                index=centered_matrix.index,
                columns=centered_matrix.index
            )
        elif self.similarity_metric == "pearson":
            # Pearson correlation
            self.similarity_matrix = centered_matrix.T.corr()
            self.similarity_matrix = self.similarity_matrix.fillna(0)
        else:
            # Euclidean distance (convert to similarity)
            distances = pairwise_distances(centered_matrix, metric='euclidean')
            # Convert distance to similarity (1 / (1 + distance))
            self.similarity_matrix = pd.DataFrame(
                1 / (1 + distances),
                index=centered_matrix.index,
                columns=centered_matrix.index
            )
        
        return self
    
    def predict(
        self,
        user_id: int,
        item_id: int,
        n_neighbors: Optional[int] = None
    ) -> float:
        """Predict rating for user-item pair"""
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix.mean().mean()  # Global mean
        
        if item_id not in self.user_item_matrix.columns:
            return self.user_mean_ratings[user_id]  # User mean
        
        # Get user's mean rating
        user_mean = self.user_mean_ratings[user_id]
        
        # Get similar users who rated this item
        similar_users = self.similarity_matrix[user_id].sort_values(ascending=False)
        similar_users = similar_users[similar_users.index != user_id]  # Exclude self
        
        # Get ratings from similar users for this item
        item_ratings = self.user_item_matrix[item_id]
        item_ratings = item_ratings.dropna()
        
        # Find intersection of similar users and users who rated the item
        common_users = similar_users.index.intersection(item_ratings.index)
        
        if len(common_users) == 0:
            return user_mean
        
        # Get top N neighbors
        top_neighbors = similar_users[common_users].head(n_neighbors)
        
        if len(top_neighbors) == 0:
            return user_mean
        
        # Calculate weighted average
        numerator = 0
        denominator = 0
        
        for neighbor_id, similarity in top_neighbors.items():
            neighbor_rating = item_ratings[neighbor_id]
            neighbor_mean = self.user_mean_ratings[neighbor_id]
            
            numerator += similarity * (neighbor_rating - neighbor_mean)
            denominator += abs(similarity)
        
        if denominator == 0:
            return user_mean
        
        prediction = user_mean + (numerator / denominator)
        # Clip to rating range [1, 5]
        prediction = np.clip(prediction, 1, 5)
        
        return prediction
    
    def recommend(
        self,
        user_id: int,
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[int, float]]:
        """Recommend items for a user"""
        if user_id not in self.user_item_matrix.index:
            # Cold start: return popular items
            item_means = self.user_item_matrix.mean().sort_values(ascending=False)
            recommendations = [
                (item_id, rating) for item_id, rating in item_means.head(n_recommendations).items()
            ]
            return recommendations
        
        # Get items user hasn't rated (or all items if exclude_rated=False)
        if exclude_rated:
            user_ratings = self.user_item_matrix.loc[user_id]
            unrated_items = user_ratings[user_ratings == 0].index.tolist()
        else:
            unrated_items = self.user_item_matrix.columns.tolist()
        
        # Predict ratings for unrated items
        predictions = []
        for item_id in unrated_items:
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]


class ItemBasedCF:
    """Item-based Collaborative Filtering"""
    
    def __init__(
        self,
        similarity_metric: str = CF_SIMILARITY_METRIC,
        n_neighbors: int = CF_NEIGHBORS
    ):
        self.similarity_metric = similarity_metric
        self.n_neighbors = n_neighbors
        self.user_item_matrix = None
        self.similarity_matrix = None
        
    def fit(self, user_item_matrix: pd.DataFrame):
        """Fit the model on user-item matrix"""
        self.user_item_matrix = user_item_matrix.copy()
        
        # Compute item-item similarity matrix
        if self.similarity_metric == "cosine":
            self.similarity_matrix = pd.DataFrame(
                cosine_similarity(self.user_item_matrix.T),
                index=self.user_item_matrix.columns,
                columns=self.user_item_matrix.columns
            )
        elif self.similarity_metric == "pearson":
            self.similarity_matrix = self.user_item_matrix.T.corr()
            self.similarity_matrix = self.similarity_matrix.fillna(0)
        else:
            # Euclidean distance
            distances = pairwise_distances(self.user_item_matrix.T, metric='euclidean')
            self.similarity_matrix = pd.DataFrame(
                1 / (1 + distances),
                index=self.user_item_matrix.columns,
                columns=self.user_item_matrix.columns
            )
        
        return self
    
    def predict(
        self,
        user_id: int,
        item_id: int,
        n_neighbors: Optional[int] = None
    ) -> float:
        """Predict rating for user-item pair"""
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix.mean().mean()  # Global mean
        
        if item_id not in self.user_item_matrix.columns:
            return self.user_item_matrix.loc[user_id].mean()  # User mean
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        user_ratings = user_ratings[user_ratings > 0]  # Only rated items
        
        if len(user_ratings) == 0:
            return self.user_item_matrix[item_id].mean()  # Item mean
        
        # Get similar items
        similar_items = self.similarity_matrix[item_id].sort_values(ascending=False)
        similar_items = similar_items[similar_items.index != item_id]  # Exclude self
        
        # Find intersection of similar items and items user rated
        common_items = similar_items.index.intersection(user_ratings.index)
        
        if len(common_items) == 0:
            return self.user_item_matrix[item_id].mean()
        
        # Get top N neighbors
        top_neighbors = similar_items[common_items].head(n_neighbors)
        
        if len(top_neighbors) == 0:
            return self.user_item_matrix[item_id].mean()
        
        # Calculate weighted average
        numerator = 0
        denominator = 0
        
        for neighbor_id, similarity in top_neighbors.items():
            neighbor_rating = user_ratings[neighbor_id]
            numerator += similarity * neighbor_rating
            denominator += abs(similarity)
        
        if denominator == 0:
            return self.user_item_matrix[item_id].mean()
        
        prediction = numerator / denominator
        prediction = np.clip(prediction, 1, 5)
        
        return prediction
    
    def recommend(
        self,
        user_id: int,
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[int, float]]:
        """Recommend items for a user"""
        if user_id not in self.user_item_matrix.index:
            # Cold start: return popular items
            item_means = self.user_item_matrix.mean().sort_values(ascending=False)
            recommendations = [
                (item_id, rating) for item_id, rating in item_means.head(n_recommendations).items()
            ]
            return recommendations
        
        # Get items user hasn't rated
        if exclude_rated:
            user_ratings = self.user_item_matrix.loc[user_id]
            unrated_items = user_ratings[user_ratings == 0].index.tolist()
        else:
            unrated_items = self.user_item_matrix.columns.tolist()
        
        # Predict ratings for unrated items
        predictions = []
        for item_id in unrated_items:
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]


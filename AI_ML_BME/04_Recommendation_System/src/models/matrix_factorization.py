"""
Matrix Factorization Models
SVD and NMF for recommendation systems
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.decomposition import NMF, TruncatedSVD
from scipy.sparse import csr_matrix

from src.config import MF_FACTORS, MF_EPOCHS, MF_REGULARIZATION, MF_LEARNING_RATE


class SVDModel:
    """Singular Value Decomposition for Matrix Factorization"""
    
    def __init__(
        self,
        n_factors: int = MF_FACTORS,
        random_state: int = 42
    ):
        self.n_factors = n_factors
        self.random_state = random_state
        self.model = None
        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, user_item_matrix: pd.DataFrame):
        """Fit SVD model"""
        self.user_item_matrix = user_item_matrix.copy()
        
        # Convert to sparse matrix
        sparse_matrix = csr_matrix(self.user_item_matrix.values)
        
        # Fit SVD
        self.model = TruncatedSVD(
            n_components=self.n_factors,
            random_state=self.random_state
        )
        self.user_factors = self.model.fit_transform(sparse_matrix)
        self.item_factors = self.model.components_.T
        
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair"""
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix.mean().mean()
        
        if item_id not in self.user_item_matrix.columns:
            return self.user_item_matrix.loc[user_id].mean()
        
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        item_idx = self.user_item_matrix.columns.get_loc(item_id)
        
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        prediction = np.clip(prediction, 1, 5)
        
        return float(prediction)
    
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
        
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_vector = self.user_factors[user_idx]
        
        # Predict for all items
        predictions = np.dot(self.item_factors, user_vector)
        
        # Create item_id to prediction mapping
        item_predictions = {
            item_id: float(pred) 
            for item_id, pred in zip(self.user_item_matrix.columns, predictions)
        }
        
        # Exclude already rated items
        if exclude_rated:
            user_ratings = self.user_item_matrix.loc[user_id]
            rated_items = user_ratings[user_ratings > 0].index
            for item_id in rated_items:
                item_predictions.pop(item_id, None)
        
        # Sort and return top N
        recommendations = sorted(
            item_predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_recommendations]
        
        return recommendations


class NMFModel:
    """Non-negative Matrix Factorization"""
    
    def __init__(
        self,
        n_factors: int = MF_FACTORS,
        random_state: int = 42,
        max_iter: int = 200
    ):
        self.n_factors = n_factors
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = None
        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, user_item_matrix: pd.DataFrame):
        """Fit NMF model"""
        self.user_item_matrix = user_item_matrix.copy()
        
        # NMF requires non-negative values, so we need to shift ratings
        # Shift from [1, 5] to [0, 4] or use implicit feedback
        matrix_shifted = self.user_item_matrix - 1  # Shift to [0, 4]
        matrix_shifted = matrix_shifted.clip(lower=0)
        
        # Convert to sparse matrix
        sparse_matrix = csr_matrix(matrix_shifted.values)
        
        # Fit NMF
        self.model = NMF(
            n_components=self.n_factors,
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        self.user_factors = self.model.fit_transform(sparse_matrix)
        self.item_factors = self.model.components_.T
        
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair"""
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix.mean().mean()
        
        if item_id not in self.user_item_matrix.columns:
            return self.user_item_matrix.loc[user_id].mean()
        
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        item_idx = self.user_item_matrix.columns.get_loc(item_id)
        
        # Predict in shifted space [0, 4]
        prediction_shifted = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        # Shift back to [1, 5]
        prediction = prediction_shifted + 1
        prediction = np.clip(prediction, 1, 5)
        
        return float(prediction)
    
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
        
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_vector = self.user_factors[user_idx]
        
        # Predict for all items
        predictions_shifted = np.dot(self.item_factors, user_vector)
        predictions = predictions_shifted + 1  # Shift back to [1, 5]
        
        # Create item_id to prediction mapping
        item_predictions = {
            item_id: float(pred) 
            for item_id, pred in zip(self.user_item_matrix.columns, predictions)
        }
        
        # Exclude already rated items
        if exclude_rated:
            user_ratings = self.user_item_matrix.loc[user_id]
            rated_items = user_ratings[user_ratings > 0].index
            for item_id in rated_items:
                item_predictions.pop(item_id, None)
        
        # Sort and return top N
        recommendations = sorted(
            item_predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_recommendations]
        
        return recommendations


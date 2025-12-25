"""
Hybrid Recommendation System
Combines multiple recommendation algorithms
"""
import numpy as np
from typing import List, Tuple, Dict, Optional

from src.config import HYBRID_WEIGHTS


class HybridRecommender:
    """Hybrid recommendation system combining multiple algorithms"""
    
    def __init__(
        self,
        models: Dict[str, any],
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize hybrid recommender
        
        Args:
            models: Dictionary of {model_name: model_object}
            weights: Dictionary of {model_name: weight} for weighted combination
        """
        self.models = models
        self.weights = weights or HYBRID_WEIGHTS
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        else:
            # Equal weights if no weights provided
            n_models = len(self.models)
            self.weights = {k: 1.0 / n_models for k in self.models.keys()}
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating using weighted combination of models"""
        predictions = []
        weights = []
        
        for model_name, model in self.models.items():
            if model_name in self.weights:
                try:
                    pred = model.predict(user_id, item_id)
                    weight = self.weights[model_name]
                    predictions.append(pred)
                    weights.append(weight)
                except Exception as e:
                    print(f"Error in {model_name} prediction: {e}")
                    continue
        
        if len(predictions) == 0:
            return 3.0  # Default neutral rating
        
        # Weighted average
        weighted_sum = sum(p * w for p, w in zip(predictions, weights))
        total_weight = sum(weights)
        
        if total_weight == 0:
            return np.mean(predictions)
        
        return weighted_sum / total_weight
    
    def recommend(
        self,
        user_id: int,
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[int, float]]:
        """Recommend items using hybrid approach"""
        # Get recommendations from all models
        all_recommendations = {}
        
        for model_name, model in self.models.items():
            if model_name in self.weights:
                try:
                    recs = model.recommend(
                        user_id=user_id,
                        n_recommendations=n_recommendations * 2,  # Get more for diversity
                        exclude_rated=exclude_rated
                    )
                    all_recommendations[model_name] = recs
                except Exception as e:
                    print(f"Error in {model_name} recommendations: {e}")
                    continue
        
        if len(all_recommendations) == 0:
            # Fallback: use first available model
            for model_name, model in self.models.items():
                try:
                    recs = model.recommend(
                        user_id=user_id,
                        n_recommendations=n_recommendations,
                        exclude_rated=exclude_rated
                    )
                    return recs[:n_recommendations]
                except:
                    continue
            return []
        
        # Combine recommendations using weighted voting
        item_scores = {}
        
        for model_name, recs in all_recommendations.items():
            weight = self.weights.get(model_name, 0.0)
            
            # Score items based on rank and weight
            for rank, (item_id, rating) in enumerate(recs, 1):
                # Higher rank = higher score, normalize by position
                score = weight * (1.0 / rank) * rating
                
                if item_id not in item_scores:
                    item_scores[item_id] = 0.0
                item_scores[item_id] += score
        
        # Sort by combined score
        recommendations = sorted(
            item_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_recommendations]
        
        # Convert to (item_id, predicted_rating) format
        # Use actual prediction for final scores
        final_recommendations = []
        for item_id, _ in recommendations:
            pred_rating = self.predict(user_id, item_id)
            final_recommendations.append((item_id, pred_rating))
        
        # Sort by predicted rating
        final_recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return final_recommendations[:n_recommendations]


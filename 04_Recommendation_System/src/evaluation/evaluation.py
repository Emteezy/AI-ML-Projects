"""
Evaluation Framework for Recommendation Models
"""
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any
from collections import defaultdict

from src.evaluation.metrics import evaluate_recommendations


class RecommendationEvaluator:
    """Evaluate recommendation models"""
    
    def __init__(self, test_ratings: pd.DataFrame, threshold: float = 3.0):
        """
        Initialize evaluator
        
        Args:
            test_ratings: DataFrame with columns ['user_id', 'item_id', 'rating']
            threshold: Minimum rating to consider item as relevant
        """
        self.test_ratings = test_ratings.copy()
        self.threshold = threshold
        
        # Convert test ratings to dictionary format
        self.test_ratings_dict = defaultdict(dict)
        for _, row in test_ratings.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            rating = row['rating']
            self.test_ratings_dict[user_id][item_id] = rating
    
    def evaluate_model(
        self,
        model: Any,
        k_values: List[int] = [5, 10, 20],
        n_recommendations: int = 20
    ) -> Dict[str, float]:
        """
        Evaluate a recommendation model
        
        Args:
            model: Model with recommend() method
            k_values: List of K values for evaluation
            n_recommendations: Number of recommendations to generate
        
        Returns:
            Dictionary of metric scores
        """
        # Generate recommendations for all test users
        recommendations = {}
        
        for user_id in self.test_ratings_dict.keys():
            try:
                rec_items = model.recommend(
                    user_id=user_id,
                    n_recommendations=n_recommendations,
                    exclude_rated=True
                )
                # Extract item IDs from (item_id, rating) tuples
                recommendations[user_id] = [item_id for item_id, _ in rec_items]
            except Exception as e:
                print(f"Error generating recommendations for user {user_id}: {e}")
                recommendations[user_id] = []
        
        # Evaluate
        results = evaluate_recommendations(
            recommendations=recommendations,
            test_ratings=self.test_ratings_dict,
            k_values=k_values,
            threshold=self.threshold
        )
        
        return results
    
    def compare_models(
        self,
        models: Dict[str, Any],
        k_values: List[int] = [5, 10, 20],
        n_recommendations: int = 20
    ) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            models: Dictionary of {model_name: model_object}
            k_values: List of K values for evaluation
            n_recommendations: Number of recommendations to generate
        
        Returns:
            DataFrame with comparison results
        """
        all_results = {}
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            results = self.evaluate_model(
                model=model,
                k_values=k_values,
                n_recommendations=n_recommendations
            )
            all_results[model_name] = results
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results).T
        return results_df


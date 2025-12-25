"""
Evaluation Metrics for Recommendation Systems
Precision@K, Recall@K, NDCG, MAP
"""
import numpy as np
from typing import List, Set, Dict
from collections import defaultdict


def precision_at_k(
    recommended_items: List[int],
    relevant_items: Set[int],
    k: int
) -> float:
    """Calculate Precision@K"""
    if k == 0:
        return 0.0
    
    recommended_k = set(recommended_items[:k])
    if len(recommended_k) == 0:
        return 0.0
    
    relevant_recommended = recommended_k.intersection(relevant_items)
    return len(relevant_recommended) / len(recommended_k)


def recall_at_k(
    recommended_items: List[int],
    relevant_items: Set[int],
    k: int
) -> float:
    """Calculate Recall@K"""
    if len(relevant_items) == 0:
        return 0.0
    
    recommended_k = set(recommended_items[:k])
    relevant_recommended = recommended_k.intersection(relevant_items)
    return len(relevant_recommended) / len(relevant_items)


def average_precision(
    recommended_items: List[int],
    relevant_items: Set[int]
) -> float:
    """Calculate Average Precision (AP)"""
    if len(relevant_items) == 0:
        return 0.0
    
    recommended_set = set(recommended_items)
    relevant_recommended = recommended_set.intersection(relevant_items)
    
    if len(relevant_recommended) == 0:
        return 0.0
    
    # Calculate precision at each position where a relevant item appears
    precisions = []
    relevant_count = 0
    
    for i, item in enumerate(recommended_items, 1):
        if item in relevant_items:
            relevant_count += 1
            precisions.append(relevant_count / i)
    
    if len(precisions) == 0:
        return 0.0
    
    return sum(precisions) / len(relevant_items)


def mean_average_precision(
    all_recommendations: Dict[int, List[int]],
    all_relevant: Dict[int, Set[int]]
) -> float:
    """Calculate Mean Average Precision (MAP)"""
    aps = []
    
    for user_id in all_recommendations:
        if user_id in all_relevant:
            ap = average_precision(
                all_recommendations[user_id],
                all_relevant[user_id]
            )
            aps.append(ap)
    
    if len(aps) == 0:
        return 0.0
    
    return np.mean(aps)


def dcg_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain at K"""
    dcg = 0.0
    recommended_k = recommended_items[:k]
    
    for i, item in enumerate(recommended_k, 1):
        if item in relevant_items:
            # Relevance is 1 if item is relevant
            dcg += 1.0 / np.log2(i + 1)
    
    return dcg


def ndcg_at_k(
    recommended_items: List[int],
    relevant_items: Set[int],
    k: int
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K"""
    dcg = dcg_at_k(recommended_items, relevant_items, k)
    
    # Ideal DCG: all relevant items ranked first
    ideal_recommended = sorted(
        list(relevant_items) + [item for item in recommended_items if item not in relevant_items],
        key=lambda x: 1 if x in relevant_items else 0,
        reverse=True
    )
    ideal_dcg = dcg_at_k(ideal_recommended, relevant_items, k)
    
    if ideal_dcg == 0:
        return 0.0
    
    return dcg / ideal_dcg


def evaluate_recommendations(
    recommendations: Dict[int, List[int]],
    test_ratings: Dict[int, Set[int]],
    k_values: List[int] = [5, 10, 20],
    threshold: float = 3.0
) -> Dict[str, float]:
    """Evaluate recommendations with multiple metrics"""
    results = {}
    
    # Convert test ratings to relevant items (ratings >= threshold)
    relevant_items = {
        user_id: {item_id for item_id, rating in ratings.items() if rating >= threshold}
        for user_id, ratings in test_ratings.items()
    }
    
    # Calculate metrics for each K
    for k in k_values:
        precisions = []
        recalls = []
        ndcgs = []
        
        for user_id in recommendations:
            if user_id in relevant_items:
                rec_items = recommendations[user_id]
                rel_items = relevant_items[user_id]
                
                precisions.append(precision_at_k(rec_items, rel_items, k))
                recalls.append(recall_at_k(rec_items, rel_items, k))
                ndcgs.append(ndcg_at_k(rec_items, rel_items, k))
        
        results[f'Precision@{k}'] = np.mean(precisions) if precisions else 0.0
        results[f'Recall@{k}'] = np.mean(recalls) if recalls else 0.0
        results[f'NDCG@{k}'] = np.mean(ndcgs) if ndcgs else 0.0
    
    # Calculate MAP
    results['MAP'] = mean_average_precision(recommendations, relevant_items)
    
    return results


"""
Train all recommendation models
"""
import argparse
import pickle
import pandas as pd
from pathlib import Path

from src.config import MODELS_DIR, DATA_DIR
from src.data.loader import MovieLensLoader
from src.data.preprocessor import DataPreprocessor
from src.models.collaborative_filtering import UserBasedCF, ItemBasedCF
from src.models.matrix_factorization import SVDModel, NMFModel
from src.models.content_based import ContentBasedFiltering
from src.models.neural_cf import NeuralCollaborativeFiltering
from src.models.hybrid import HybridRecommender
from src.evaluation.evaluation import RecommendationEvaluator


def train_models(
    dataset_size: str = "100k",
    train_models_list: list = None,
    evaluate: bool = True
):
    """Train all recommendation models"""
    
    print("=" * 60)
    print("Movie Recommendation System - Model Training")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading dataset...")
    loader = MovieLensLoader(dataset_size=dataset_size)
    loader.download_dataset()
    ratings, movies, users = loader.load_all()
    print(f"   Loaded {len(ratings)} ratings, {len(movies)} movies")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    ratings_filtered = preprocessor.filter_sparse_data(ratings)
    ratings_encoded, mappings = preprocessor.encode_ids(ratings_filtered)
    
    # Split data
    train_ratings, test_ratings = preprocessor.split_data(ratings_encoded)
    print(f"   Train: {len(train_ratings)} ratings, Test: {len(test_ratings)} ratings")
    
    # Create user-item matrix
    user_item_matrix = preprocessor.create_user_item_matrix(
        train_ratings,
        user_col='user_idx',
        item_col='item_idx',
        rating_col='rating'
    )
    print(f"   User-item matrix: {user_item_matrix.shape}")
    
    # Map back to original IDs for movies
    movies_mapped = movies.copy()
    idx_to_item = mappings['idx_to_item']
    movies_mapped['item_idx'] = movies_mapped['movie_id'].map(
        {v: k for k, v in mappings['item_to_idx'].items()}
    )
    movies_mapped = movies_mapped.dropna(subset=['item_idx'])
    movies_mapped['item_idx'] = movies_mapped['item_idx'].astype(int)
    
    # Train models
    models = {}
    train_models_list = train_models_list or [
        'user_cf', 'item_cf', 'svd', 'nmf', 'content_based', 'neural_cf', 'hybrid'
    ]
    
    print("\n3. Training models...")
    
    # User-based CF
    if 'user_cf' in train_models_list:
        print("   Training User-based Collaborative Filtering...")
        user_cf = UserBasedCF()
        user_cf.fit(user_item_matrix)
        models['user_cf'] = user_cf
        # Save model
        with open(MODELS_DIR / 'user_cf_model.pkl', 'wb') as f:
            pickle.dump(user_cf, f)
        print("   ✓ User-based CF trained and saved")
    
    # Item-based CF
    if 'item_cf' in train_models_list:
        print("   Training Item-based Collaborative Filtering...")
        item_cf = ItemBasedCF()
        item_cf.fit(user_item_matrix)
        models['item_cf'] = item_cf
        with open(MODELS_DIR / 'item_cf_model.pkl', 'wb') as f:
            pickle.dump(item_cf, f)
        print("   ✓ Item-based CF trained and saved")
    
    # SVD
    if 'svd' in train_models_list:
        print("   Training SVD...")
        svd = SVDModel()
        svd.fit(user_item_matrix)
        models['svd'] = svd
        with open(MODELS_DIR / 'svd_model.pkl', 'wb') as f:
            pickle.dump(svd, f)
        print("   ✓ SVD trained and saved")
    
    # NMF
    if 'nmf' in train_models_list:
        print("   Training NMF...")
        nmf = NMFModel()
        nmf.fit(user_item_matrix)
        models['nmf'] = nmf
        with open(MODELS_DIR / 'nmf_model.pkl', 'wb') as f:
            pickle.dump(nmf, f)
        print("   ✓ NMF trained and saved")
    
    # Content-based
    if 'content_based' in train_models_list:
        print("   Training Content-based Filtering...")
        content_based = ContentBasedFiltering()
        content_based.fit(user_item_matrix, movies_mapped)
        models['content_based'] = content_based
        with open(MODELS_DIR / 'content_based_model.pkl', 'wb') as f:
            pickle.dump(content_based, f)
        print("   ✓ Content-based trained and saved")
    
    # Neural CF
    if 'neural_cf' in train_models_list:
        print("   Training Neural Collaborative Filtering...")
        neural_cf = NeuralCollaborativeFiltering(epochs=5)  # Reduced for faster training
        neural_cf.fit(user_item_matrix)
        models['neural_cf'] = neural_cf
        with open(MODELS_DIR / 'neural_cf_model.pkl', 'wb') as f:
            pickle.dump(neural_cf, f)
        print("   ✓ Neural CF trained and saved")
    
    # Hybrid
    if 'hybrid' in train_models_list:
        print("   Training Hybrid Recommender...")
        # Use available models
        available_models = {k: v for k, v in models.items() if k != 'hybrid'}
        if len(available_models) > 0:
            hybrid = HybridRecommender(available_models)
            models['hybrid'] = hybrid
            with open(MODELS_DIR / 'hybrid_model.pkl', 'wb') as f:
                pickle.dump(hybrid, f)
            print("   ✓ Hybrid recommender trained and saved")
    
    # Evaluation
    if evaluate and len(test_ratings) > 0:
        print("\n4. Evaluating models...")
        evaluator = RecommendationEvaluator(test_ratings)
        
        # Convert test ratings to use idx
        test_ratings_idx = test_ratings.copy()
        
        # Evaluate each model
        results = {}
        for model_name, model in models.items():
            if model_name == 'hybrid':
                continue  # Skip hybrid for now
            print(f"   Evaluating {model_name}...")
            try:
                # Create a wrapper that uses idx
                class ModelWrapper:
                    def __init__(self, model, user_item_matrix):
                        self.model = model
                        self.user_item_matrix = user_item_matrix
                    
                    def recommend(self, user_id, n_recommendations, exclude_rated):
                        # Convert user_id to idx if needed
                        return self.model.recommend(user_id, n_recommendations, exclude_rated)
                
                wrapper = ModelWrapper(model, user_item_matrix)
                eval_results = evaluator.evaluate_model(wrapper, k_values=[5, 10, 20])
                results[model_name] = eval_results
                print(f"     Precision@10: {eval_results.get('Precision@10', 0):.4f}")
            except Exception as e:
                print(f"     Error evaluating {model_name}: {e}")
        
        # Save results
        if results:
            results_df = pd.DataFrame(results).T
            results_path = Path(__file__).parent.parent.parent / "results" / "evaluation_results.csv"
            results_df.to_csv(results_path)
            print(f"\n   Evaluation results saved to {results_path}")
            print("\n" + str(results_df))
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nModels saved to: {MODELS_DIR}")
    print(f"Available models: {list(models.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Train recommendation models")
    parser.add_argument(
        "--dataset-size",
        type=str,
        default="100k",
        choices=["100k", "1m", "10m"],
        help="Dataset size"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=['user_cf', 'item_cf', 'svd', 'nmf', 'content_based', 'neural_cf', 'hybrid'],
        help="Models to train (default: all)"
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation"
    )
    
    args = parser.parse_args()
    
    train_models(
        dataset_size=args.dataset_size,
        train_models_list=args.models,
        evaluate=not args.no_eval
    )


if __name__ == "__main__":
    main()


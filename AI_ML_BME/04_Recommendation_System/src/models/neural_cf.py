"""
Neural Collaborative Filtering
Deep learning model for recommendations
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.config import (
    NCF_EMBEDDING_DIM, NCF_LAYERS, NCF_EPOCHS, 
    NCF_BATCH_SIZE, NCF_LEARNING_RATE
)


class NCFDataset(Dataset):
    """Dataset for Neural Collaborative Filtering"""
    
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


class NeuralCF(nn.Module):
    """Neural Collaborative Filtering Model"""
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = NCF_EMBEDDING_DIM,
        layers: List[int] = NCF_LAYERS
    ):
        super(NeuralCF, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # MLP layers
        mlp_layers = []
        input_dim = embedding_dim * 2  # Concatenated user and item embeddings
        
        for output_dim in layers:
            mlp_layers.append(nn.Linear(input_dim, output_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            input_dim = output_dim
        
        # Output layer
        mlp_layers.append(nn.Linear(input_dim, 1))
        mlp_layers.append(nn.Sigmoid())  # Output in [0, 1], will scale to [1, 5]
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        concat = torch.cat([user_emb, item_emb], dim=1)
        
        # Pass through MLP
        output = self.mlp(concat)
        
        return output.squeeze()


class NeuralCollaborativeFiltering:
    """Neural Collaborative Filtering Model Wrapper"""
    
    def __init__(
        self,
        embedding_dim: int = NCF_EMBEDDING_DIM,
        layers: List[int] = NCF_LAYERS,
        epochs: int = NCF_EPOCHS,
        batch_size: int = NCF_BATCH_SIZE,
        learning_rate: float = NCF_LEARNING_RATE,
        device: str = 'cpu'
    ):
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.user_item_matrix = None
        self.user_to_idx = None
        self.item_to_idx = None
        self.idx_to_user = None
        self.idx_to_item = None
        self.n_users = 0
        self.n_items = 0
    
    def fit(self, user_item_matrix: pd.DataFrame):
        """Train Neural CF model"""
        self.user_item_matrix = user_item_matrix.copy()
        
        # Create mappings
        unique_users = sorted(user_item_matrix.index.unique())
        unique_items = sorted(user_item_matrix.columns.unique())
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        
        # Prepare training data
        train_data = []
        for user_id in user_item_matrix.index:
            for item_id in user_item_matrix.columns:
                rating = user_item_matrix.loc[user_id, item_id]
                if rating > 0:  # Only include rated items
                    user_idx = self.user_to_idx[user_id]
                    item_idx = self.item_to_idx[item_id]
                    # Normalize rating to [0, 1] range
                    rating_norm = (rating - 1) / 4.0
                    train_data.append((user_idx, item_idx, rating_norm))
        
        if len(train_data) == 0:
            raise ValueError("No training data found")
        
        # Create dataset and dataloader
        user_ids, item_ids, ratings = zip(*train_data)
        dataset = NCFDataset(user_ids, item_ids, ratings)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = NeuralCF(
            n_users=self.n_users,
            n_items=self.n_items,
            embedding_dim=self.embedding_dim,
            layers=self.layers
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for user_batch, item_batch, rating_batch in dataloader:
                user_batch = user_batch.to(self.device)
                item_batch = item_batch.to(self.device)
                rating_batch = rating_batch.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(user_batch, item_batch)
                loss = criterion(predictions, rating_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair"""
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return self.user_item_matrix.mean().mean()
        
        self.model.eval()
        with torch.no_grad():
            user_idx = torch.LongTensor([self.user_to_idx[user_id]]).to(self.device)
            item_idx = torch.LongTensor([self.item_to_idx[item_id]]).to(self.device)
            
            prediction_norm = self.model(user_idx, item_idx).item()
            # Denormalize from [0, 1] to [1, 5]
            prediction = prediction_norm * 4.0 + 1.0
            prediction = np.clip(prediction, 1, 5)
        
        return float(prediction)
    
    def recommend(
        self,
        user_id: int,
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[int, float]]:
        """Recommend items for a user"""
        if user_id not in self.user_to_idx:
            # Cold start: return popular items
            item_means = self.user_item_matrix.mean().sort_values(ascending=False)
            recommendations = [
                (item_id, rating) for item_id, rating in item_means.head(n_recommendations).items()
            ]
            return recommendations
        
        # Get items to predict
        if exclude_rated:
            user_ratings = self.user_item_matrix.loc[user_id]
            unrated_items = user_ratings[user_ratings == 0].index.tolist()
        else:
            unrated_items = self.user_item_matrix.columns.tolist()
        
        # Predict for all unrated items
        predictions = []
        user_idx = self.user_to_idx[user_id]
        user_idx_tensor = torch.LongTensor([user_idx]).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            for item_id in unrated_items:
                if item_id in self.item_to_idx:
                    item_idx = torch.LongTensor([self.item_to_idx[item_id]]).to(self.device)
                    prediction_norm = self.model(user_idx_tensor, item_idx).item()
                    prediction = prediction_norm * 4.0 + 1.0
                    predictions.append((item_id, float(prediction)))
        
        # Sort and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]


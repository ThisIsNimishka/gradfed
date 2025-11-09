"""
Gradient Monitor for tracking client gradient quality and stability
Place this file in: src/gradient_monitor.py
"""

import torch
import numpy as np
from collections import deque
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class GradientMonitor:
    """
    Tracks gradient consistency and quality for each client over time.
    Used to identify noisy/low-quality clients in federated learning.
    """
    
    def __init__(self, window_size=5, quality_threshold=0.3, 
                 stability_weight=0.5, magnitude_weight=0.3, consistency_weight=0.2):
        """
        Args:
            window_size: Number of recent gradients to track per client
            quality_threshold: Threshold below which client is considered noisy
            stability_weight: Weight for gradient stability in quality score
            magnitude_weight: Weight for gradient magnitude in quality score
            consistency_weight: Weight for gradient consistency in quality score
        """
        self.window_size = window_size
        self.quality_threshold = quality_threshold
        self.stability_weight = stability_weight
        self.magnitude_weight = magnitude_weight
        self.consistency_weight = consistency_weight
        
        # Storage
        self.gradient_norms = {}        # client_id -> deque of gradient norms
        self.gradient_cosines = {}      # client_id -> deque of cosine similarities
        self.quality_scores = {}        # client_id -> current quality score
        self.selection_count = {}       # client_id -> number of times selected
        self.contribution_history = {}  # client_id -> historical contribution quality
        self._stored_gradients = {}     # client_id -> previous gradient for comparison
        
    def update(self, client_id: int, local_weights: Dict[str, torch.Tensor], 
               global_weights: Dict[str, torch.Tensor]) -> float:
        """
        Update gradient statistics for a client.
        
        Args:
            client_id: Client identifier
            local_weights: Client's updated model weights
            global_weights: Global model weights before update
            
        Returns:
            quality_score: Current quality score for this client (0-1)
        """
        # Calculate pseudo-gradients (weight differences)
        gradients = {}
        for key in local_weights.keys():
            if key in global_weights:
                gradients[key] = global_weights[key].cpu() - local_weights[key].cpu()
        
        # Initialize storage for new client
        if client_id not in self.gradient_norms:
            self.gradient_norms[client_id] = deque(maxlen=self.window_size)
            self.gradient_cosines[client_id] = deque(maxlen=self.window_size)
            self.quality_scores[client_id] = 1.0  # Start with high quality
            self.selection_count[client_id] = 0
            self.contribution_history[client_id] = []
        
        # Calculate gradient norm
        grad_norm = self._calculate_gradient_norm(gradients)
        self.gradient_norms[client_id].append(grad_norm)
        
        # Calculate cosine similarity with previous gradient (if exists)
        if client_id in self._stored_gradients:
            prev_grad = self._stored_gradients[client_id]
            cosine_sim = self._calculate_cosine_similarity(gradients, prev_grad)
            self.gradient_cosines[client_id].append(cosine_sim)
        
        # Store current gradient for next comparison
        self._store_gradient(client_id, gradients)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(client_id)
        self.quality_scores[client_id] = quality_score
        
        # Update selection count
        self.selection_count[client_id] += 1
        
        # Store in contribution history
        self.contribution_history[client_id].append(quality_score)
        
        return quality_score
    
    def _calculate_gradient_norm(self, gradients: Dict[str, torch.Tensor]) -> float:
        """Calculate L2 norm of gradients"""
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += torch.sum(grad ** 2).item()
        return np.sqrt(total_norm)
    
    def _calculate_cosine_similarity(self, grad1: Dict[str, torch.Tensor], 
                                    grad2: Dict[str, torch.Tensor]) -> float:
        """Calculate cosine similarity between two gradient dictionaries"""
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0
        
        for key in grad1.keys():
            if key in grad2:
                dot_product += torch.sum(grad1[key] * grad2[key]).item()
                norm1 += torch.sum(grad1[key] ** 2).item()
                norm2 += torch.sum(grad2[key] ** 2).item()
        
        norm1 = np.sqrt(norm1)
        norm2 = np.sqrt(norm2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _store_gradient(self, client_id: int, gradients: Dict[str, torch.Tensor]):
        """Store gradient for future comparison (memory efficient)"""
        # Store as a flattened tensor to save memory
        flat_grad = torch.cat([g.flatten() for g in gradients.values()])
        self._stored_gradients[client_id] = {'flat': flat_grad}
    
    def _calculate_quality_score(self, client_id: int) -> float:
        """
        Calculate comprehensive quality score based on:
        1. Gradient stability (low variance)
        2. Gradient magnitude (not too small/large)
        3. Gradient consistency (positive cosine similarity)
        """
        score = 0.0
        
        # 1. Stability score (inverse of coefficient of variation)
        if len(self.gradient_norms[client_id]) >= 2:
            norms = list(self.gradient_norms[client_id])
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)
            
            if mean_norm > 1e-8:
                cv = std_norm / mean_norm
                # Lower CV = more stable = higher score
                stability_score = 1.0 / (1.0 + cv)
            else:
                stability_score = 0.1  # Very small gradients
            
            score += self.stability_weight * stability_score
        else:
            # New client - neutral score
            score += self.stability_weight * 0.5
        
        # 2. Magnitude score (prefer moderate gradients)
        if len(self.gradient_norms[client_id]) > 0:
            recent_norm = self.gradient_norms[client_id][-1]
            # Use sigmoid-like function to normalize
            # Penalize both very small and very large gradients
            magnitude_score = 2 / (1 + np.exp(recent_norm / 10.0))
            score += self.magnitude_weight * magnitude_score
        else:
            score += self.magnitude_weight * 0.5
        
        # 3. Consistency score (cosine similarity)
        if len(self.gradient_cosines[client_id]) >= 1:
            avg_cosine = np.mean(list(self.gradient_cosines[client_id]))
            # Map from [-1, 1] to [0, 1]
            consistency_score = (avg_cosine + 1) / 2
            score += self.consistency_weight * consistency_score
        else:
            # New client - neutral score
            score += self.consistency_weight * 0.5
        
        return np.clip(score, 0.0, 1.0)
    
    def get_quality_score(self, client_id: int) -> float:
        """Get current quality score for a client"""
        return self.quality_scores.get(client_id, 0.5)  # Default neutral
    
    def is_noisy(self, client_id: int) -> bool:
        """Check if client is considered noisy based on quality threshold"""
        return self.quality_scores.get(client_id, 1.0) < self.quality_threshold
    
    def get_all_quality_scores(self) -> Dict[int, float]:
        """Get quality scores for all tracked clients"""
        return self.quality_scores.copy()
    
    def get_statistics(self) -> Dict:
        """Get monitoring statistics"""
        if not self.quality_scores:
            return {
                'avg_quality': 0.5,
                'std_quality': 0.0,
                'min_quality': 0.0,
                'max_quality': 1.0,
                'num_noisy': 0,
                'total_clients': 0
            }
        
        quality_values = list(self.quality_scores.values())
        return {
            'avg_quality': np.mean(quality_values),
            'std_quality': np.std(quality_values),
            'min_quality': np.min(quality_values),
            'max_quality': np.max(quality_values),
            'num_noisy': sum(1 for q in quality_values if q < self.quality_threshold),
            'total_clients': len(quality_values)
        }
    
    def get_client_history(self, client_id: int) -> Dict:
        """Get detailed history for a client"""
        return {
            'quality_score': self.quality_scores.get(client_id, None),
            'gradient_norms': list(self.gradient_norms.get(client_id, [])),
            'cosine_similarities': list(self.gradient_cosines.get(client_id, [])),
            'selection_count': self.selection_count.get(client_id, 0),
            'contribution_history': self.contribution_history.get(client_id, [])
        }
    
    def reset_client(self, client_id: int):
        """Reset tracking for a specific client"""
        if client_id in self.gradient_norms:
            del self.gradient_norms[client_id]
        if client_id in self.gradient_cosines:
            del self.gradient_cosines[client_id]
        if client_id in self.quality_scores:
            del self.quality_scores[client_id]
        if client_id in self.selection_count:
            del self.selection_count[client_id]
        if client_id in self.contribution_history:
            del self.contribution_history[client_id]
        if client_id in self._stored_gradients:
            del self._stored_gradients[client_id]
    
    def reset_all(self):
        """Reset all tracking data"""
        self.gradient_norms.clear()
        self.gradient_cosines.clear()
        self.quality_scores.clear()
        self.selection_count.clear()
        self.contribution_history.clear()
        self._stored_gradients.clear()
    
    def get_top_quality_clients(self, n: int = 10) -> List[Tuple[int, float]]:
        """Get top n clients by quality score"""
        sorted_clients = sorted(self.quality_scores.items(), 
                              key=lambda x: x[1], reverse=True)
        return sorted_clients[:n]
    
    def get_noisy_clients(self) -> List[int]:
        """Get list of client IDs considered noisy"""
        return [client_id for client_id, score in self.quality_scores.items()
                if score < self.quality_threshold]
    
    def print_summary(self):
        """Print summary of gradient monitoring"""
        stats = self.get_statistics()
        
        logger.info("\n" + "="*60)
        logger.info("Gradient Monitor Summary")
        logger.info("="*60)
        logger.info(f"Total Clients Tracked: {stats['total_clients']}")
        logger.info(f"Average Quality Score: {stats['avg_quality']:.3f}")
        logger.info(f"Quality Std Dev: {stats['std_quality']:.3f}")
        logger.info(f"Quality Range: [{stats['min_quality']:.3f}, {stats['max_quality']:.3f}]")
        logger.info(f"Noisy Clients (< {self.quality_threshold}): {stats['num_noisy']}")
        
        if self.quality_scores:
            top_clients = self.get_top_quality_clients(5)
            logger.info("\nTop 5 Quality Clients:")
            for client_id, score in top_clients:
                logger.info(f"  Client {client_id}: {score:.3f}")
            
            noisy = self.get_noisy_clients()
            if noisy:
                logger.info(f"\nNoisy Clients: {noisy[:10]}")  # Show first 10
        
        logger.info("="*60 + "\n")
"""
Modified PPO Update with Gradient-Aware Client Selection
Place this file in: src/ppo_server/gradfed_ppo_update.py
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import copy
import logging

logger = logging.getLogger(__name__)


class GradFedPPO:
    """
    Enhanced PPO-based federated learning with gradient-aware client selection.
    Integrates GradientMonitor to identify and deprioritize noisy clients.
    """
    
    def __init__(self, args, dataset_train, dataset_test, dict_users, net_glob, gradient_monitor):
        """
        Args:
            args: Command line arguments
            dataset_train: Training dataset
            dataset_test: Test dataset
            dict_users: Dictionary mapping user_id to their data indices
            net_glob: Global model
            gradient_monitor: GradientMonitor instance for tracking gradient quality
        """
        self.args = args
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dict_users = dict_users
        self.net_glob = net_glob
        self.gradient_monitor = gradient_monitor
        
        # PPO components (will be initialized by PPO actor)
        self.ppo_actor = None
        
        # Tracking
        self.client_participation = {i: 0 for i in range(args.num_users)}
        self.round_accuracies = []
        self.round_losses = []
        
    def set_ppo_actor(self, ppo_actor):
        """Set PPO actor from main script"""
        self.ppo_actor = ppo_actor
    
    def select_clients_with_gradient_awareness(self, state, available_clients, 
                                               ppo_probs, exploration_rate=0.1):
        """
        Enhanced client selection combining PPO policy with gradient quality scores.
        
        Args:
            state: Current environment state
            available_clients: List of available client IDs
            ppo_probs: PPO action probabilities for each client
            exploration_rate: Probability of exploring low-quality clients
            
        Returns:
            selected_clients: List of selected client IDs
            selection_scores: Dictionary of selection scores for logging
        """
        num_selected = max(int(self.args.frac * self.args.num_users), 1)
        
        # Get gradient quality scores
        quality_scores = {}
        for client_id in available_clients:
            quality_scores[client_id] = self.gradient_monitor.get_quality_score(client_id)
        
        # Combine PPO probabilities with quality scores
        combined_scores = {}
        for client_id in available_clients:
            ppo_score = ppo_probs[client_id] if client_id < len(ppo_probs) else 0.5
            quality_score = quality_scores[client_id]
            
            # Weighted combination (adjust alpha to balance PPO vs quality)
            alpha = 0.7  # Weight for PPO
            beta = 0.3   # Weight for gradient quality
            combined_scores[client_id] = alpha * ppo_score + beta * quality_score
        
        # Exploration: occasionally select potentially noisy clients for learning
        if np.random.random() < exploration_rate:
            # Select some low-quality clients to see if they improve
            low_quality_clients = [c for c in available_clients 
                                  if quality_scores[c] < 0.4]
            if low_quality_clients:
                explore_count = max(1, num_selected // 5)  # 20% exploration
                explored = np.random.choice(low_quality_clients, 
                                          min(explore_count, len(low_quality_clients)),
                                          replace=False).tolist()
                
                # Select rest based on combined scores
                remaining = num_selected - len(explored)
                remaining_clients = [c for c in available_clients if c not in explored]
                
                if remaining > 0 and remaining_clients:
                    remaining_scores = {c: combined_scores[c] for c in remaining_clients}
                    sorted_clients = sorted(remaining_scores.items(), 
                                          key=lambda x: x[1], reverse=True)
                    exploited = [c for c, _ in sorted_clients[:remaining]]
                else:
                    exploited = []
                
                selected_clients = explored + exploited
                logger.info(f"Exploration mode: {len(explored)} low-quality, {len(exploited)} high-quality")
            else:
                # No low-quality clients, use normal selection
                selected_clients = self._select_top_k(combined_scores, num_selected)
        else:
            # Exploitation: select based on combined scores
            selected_clients = self._select_top_k(combined_scores, num_selected)
        
        # Update participation tracking
        for client_id in selected_clients:
            self.client_participation[client_id] += 1
        
        # Prepare selection info for logging
        selection_info = {
            'selected': selected_clients,
            'quality_scores': {c: quality_scores[c] for c in selected_clients},
            'combined_scores': {c: combined_scores[c] for c in selected_clients},
            'avg_quality': np.mean([quality_scores[c] for c in selected_clients]),
            'num_noisy': sum(1 for c in selected_clients if quality_scores[c] < 0.3)
        }
        
        return selected_clients, selection_info
    
    def _select_top_k(self, scores_dict, k):
        """Select top k clients based on scores"""
        sorted_clients = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        return [client_id for client_id, _ in sorted_clients[:k]]
    
    def aggregate_with_quality_weights(self, client_weights, selected_clients, 
                                      data_sizes):
        """
        Aggregate client models weighted by both data size and gradient quality.
        
        Args:
            client_weights: List of client model weights
            selected_clients: List of selected client IDs
            data_sizes: Dictionary of data sizes per client
            
        Returns:
            aggregated_weights: Weighted aggregated model weights
        """
        # Get quality scores
        quality_scores = [self.gradient_monitor.get_quality_score(c) 
                         for c in selected_clients]
        
        # Calculate weights combining data size and quality
        total_data = sum(data_sizes[c] for c in selected_clients)
        total_quality = sum(quality_scores)
        
        if total_quality == 0:
            # Fallback to data-size-only weighting
            weights = [data_sizes[c] / total_data for c in selected_clients]
        else:
            # Combined weighting: 60% data size, 40% quality
            data_weights = [data_sizes[c] / total_data for c in selected_clients]
            quality_weights = [q / total_quality for q in quality_scores]
            weights = [0.6 * dw + 0.4 * qw 
                      for dw, qw in zip(data_weights, quality_weights)]
            
            # Normalize
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        
        # Aggregate
        aggregated = {}
        for key in client_weights[0].keys():
            aggregated[key] = sum(w * client_weights[i][key] 
                                 for i, w in enumerate(weights))
        
        logger.info(f"Aggregation weights - Data: {data_weights}, Quality: {quality_weights}, Final: {weights}")
        
        return aggregated
    
    def update_gradient_monitor(self, client_id, local_weights, global_weights):
        """
        Update gradient monitor after client training.
        
        Args:
            client_id: Client identifier
            local_weights: Client's updated model weights
            global_weights: Global model weights before update
            
        Returns:
            quality_score: Updated quality score for the client
        """
        quality_score = self.gradient_monitor.update(client_id, local_weights, global_weights)
        
        if quality_score < 0.3:
            logger.warning(f"Client {client_id} has low quality score: {quality_score:.3f}")
        
        return quality_score
    
    def calculate_reward_with_quality(self, prev_accuracy, curr_accuracy, 
                                     prev_loss, curr_loss, selected_clients,
                                     comm_cost):
        """
        Calculate PPO reward incorporating gradient quality metrics.
        
        Args:
            prev_accuracy: Previous round accuracy
            curr_accuracy: Current round accuracy
            prev_loss: Previous round loss
            curr_loss: Current round loss
            selected_clients: List of selected client IDs
            comm_cost: Communication cost
            
        Returns:
            reward: Calculated reward value
        """
        # Accuracy improvement (primary objective)
        acc_gain = (curr_accuracy - prev_accuracy) * 100
        
        # Loss reduction
        loss_reduction = (prev_loss - curr_loss) * 10
        
        # Quality bonus: reward selecting high-quality clients
        quality_scores = [self.gradient_monitor.get_quality_score(c) 
                         for c in selected_clients]
        avg_quality = np.mean(quality_scores) if quality_scores else 0.5
        quality_bonus = avg_quality * 5
        
        # Diversity bonus: encourage exploring different clients
        selection_variance = np.var([self.client_participation[c] 
                                    for c in selected_clients])
        diversity_bonus = min(selection_variance / 10, 2.0)
        
        # Communication penalty
        comm_penalty = comm_cost * 0.1
        
        # Total reward
        reward = acc_gain + loss_reduction + quality_bonus + diversity_bonus - comm_penalty
        
        logger.info(f"Reward components - Acc: {acc_gain:.2f}, Loss: {loss_reduction:.2f}, "
                   f"Quality: {quality_bonus:.2f}, Diversity: {diversity_bonus:.2f}, "
                   f"Comm: -{comm_penalty:.2f}, Total: {reward:.2f}")
        
        return reward
    
    def get_enhanced_state(self, base_state, round_num):
        """
        Enhance PPO state with gradient quality features.
        
        Args:
            base_state: Original PPO state
            round_num: Current round number
            
        Returns:
            enhanced_state: State with additional gradient quality features
        """
        stats = self.gradient_monitor.get_statistics()
        
        # Additional features
        quality_features = [
            stats['avg_quality'],
            stats['num_noisy'] / max(stats['total_clients'], 1),
            stats.get('std_quality', 0.0)
        ]
        
        # Concatenate with base state
        if isinstance(base_state, np.ndarray):
            enhanced_state = np.concatenate([base_state, quality_features])
        elif isinstance(base_state, list):
            enhanced_state = base_state + quality_features
        else:
            enhanced_state = list(base_state) + quality_features
        
        return np.array(enhanced_state, dtype=np.float32)
    
    def log_round_statistics(self, round_num, selected_clients, accuracy, loss):
        """Log statistics for current round"""
        stats = self.gradient_monitor.get_statistics()
        
        logger.info(f"\n=== Round {round_num} Statistics ===")
        logger.info(f"Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        logger.info(f"Selected clients: {selected_clients}")
        logger.info(f"Gradient Quality - Avg: {stats['avg_quality']:.3f}, "
                   f"Noisy: {stats['num_noisy']}/{stats['total_clients']}")
        
        # Log per-client quality
        for client_id in selected_clients:
            q_score = self.gradient_monitor.get_quality_score(client_id)
            logger.info(f"  Client {client_id}: Quality={q_score:.3f}, "
                       f"Participations={self.client_participation[client_id]}")
    
    def save_statistics(self, filepath):
        """Save training statistics to file"""
        stats = {
            'round_accuracies': self.round_accuracies,
            'round_losses': self.round_losses,
            'client_participation': self.client_participation,
            'quality_scores': self.gradient_monitor.get_all_quality_scores(),
            'gradient_stats': self.gradient_monitor.get_statistics()
        }
        
        np.savez(filepath, **stats)
        logger.info(f"Statistics saved to {filepath}")
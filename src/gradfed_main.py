"""
GradFed Main Training Script
Place this file in: src/gradfed_main.py

Run with:
python src/gradfed_main.py --dataset cifar10 --model resnet10 --num_users 100 \
    --frac 0.1 --local_ep 10 --local_bs 10 --epochs 100 --gpu 0
"""

import sys
import os
import copy
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Import from existing files
from utils import get_dataset, exp_details
from options import args_parser
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, ResNet10Cifar, ResNet10Emnist
from sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from gradient_monitor import GradientMonitor
from ppo_server.gradfed_ppo_update import GradFedPPO
from ppo_server.ppo import PPO

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gradfed_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class"""
    
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate:
    """Local client training"""
    
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs), 
                                      batch_size=self.args.local_bs, 
                                      shuffle=True)
        self.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() 
                                   and args.gpu_id != -1 else 'cpu')
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
    
    def update_weights(self, model, global_round):
        """Train local model and return updated weights"""
        model.train()
        epoch_loss = []
        
        # Set optimizer
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                       momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                        weight_decay=1e-4)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                       momentum=0.5)
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                
                batch_loss.append(loss.item())
            
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


def test_inference(args, model, test_dataset):
    """Returns the test accuracy and loss"""
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() 
                         and args.gpu_id != -1 else 'cpu')
    criterion = torch.nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            
            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            
            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    
    accuracy = correct/total
    avg_loss = loss/len(testloader)
    return accuracy, avg_loss


def gradfed_train():
    """Main GradFed training function"""
    
    start_time = time.time()
    
    # Parse arguments
    args = args_parser()
    exp_details(args)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() 
                         and args.gpu_id != -1 else 'cpu')
    
    # Load dataset and split users
    train_dataset, test_dataset, dict_users = get_dataset(args)
    
    # Build model
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
            global_model = CNNCifar(args=args)
        else:
            exit('Error: unrecognized dataset')
    elif args.model == 'resnet10':
        if args.dataset == "emnist":
            global_model = ResNet10Emnist(args=args)
        else:
            global_model = ResNet10Cifar(args=args)
    elif args.model == 'mlp':
        len_in = 1
        for x in train_dataset[0][0].shape:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64, 
                          dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')
    
    # Set the model to train and send to device
    global_model.to(device)
    global_model.train()
    logger.info(f"Model: {args.model}, Device: {device}")
    
    # Copy weights
    global_weights = global_model.state_dict()
    
    # Initialize GradientMonitor
    gradient_monitor = GradientMonitor(
        window_size=5,
        quality_threshold=0.3,
        stability_weight=0.5,
        magnitude_weight=0.3,
        consistency_weight=0.2
    )
    logger.info("GradientMonitor initialized")
    
    # Initialize PPO Actor
    # State dimension: [accuracy, loss, round_progress] + gradient_quality_features
    state_dim = 3 + 3  # 3 base + 3 quality features
    action_dim = args.num_users
    
    ppo = PPO(
        args=args,
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        K_epochs=4,
        eps_clip=0.2,
        has_continuous_action_space=False
    )
    logger.info(f"PPO initialized - State dim: {state_dim}, Action dim: {action_dim}")
    
    # Initialize GradFedPPO
    gradfed_ppo = GradFedPPO(
        args=args,
        dataset_train=train_dataset,
        dataset_test=test_dataset,
        dict_users=dict_users,
        net_glob=global_model,
        gradient_monitor=gradient_monitor
    )
    gradfed_ppo.set_ppo_actor(ppo)
    logger.info("GradFedPPO initialized")
    
    # Training
    train_loss, train_accuracy = [], []
    print_every = 1
    
    # Initial evaluation
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    logger.info(f"Initial Test Accuracy: {100*test_acc:.2f}%")
    
    prev_accuracy = test_acc
    prev_loss = test_loss
    
    for epoch in range(args.epochs):
        local_weights, local_losses = [], []
        logger.info(f'\n | Global Training Round: {epoch+1}/{args.epochs} |\n')
        
        global_model.train()
        
        # Construct state for PPO
        base_state = [
            prev_accuracy,
            prev_loss,
            epoch / args.epochs  # Progress
        ]
        
        # Enhance state with gradient quality features
        state = gradfed_ppo.get_enhanced_state(base_state, epoch)
        state_tensor = torch.FloatTensor(state).to(device)
        
        # Get PPO action (client selection probabilities)
        with torch.no_grad():
            action_probs = ppo.policy_old.actor(state_tensor).cpu().numpy()
        
        # Select clients with gradient awareness
        available_clients = list(range(args.num_users))
        selected_clients, selection_info = gradfed_ppo.select_clients_with_gradient_awareness(
            state=state_tensor,
            available_clients=available_clients,
            ppo_probs=action_probs,
            exploration_rate=max(0.1, 0.3 - epoch/args.epochs*0.2)  # Decay exploration
        )
        
        logger.info(f"Selected {len(selected_clients)} clients: {selected_clients}")
        logger.info(f"Avg quality score: {selection_info['avg_quality']:.3f}")
        logger.info(f"Noisy clients: {selection_info['num_noisy']}/{len(selected_clients)}")
        
        # Store global weights before update (for gradient calculation)
        global_weights_before = copy.deepcopy(global_model.state_dict())
        
        # Local training for selected clients
        data_sizes = {}
        for idx in selected_clients:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                     idxs=dict_users[idx])
            
            # Train local model
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch
            )
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            data_sizes[idx] = len(dict_users[idx])
            
            # Update gradient monitor
            quality_score = gradfed_ppo.update_gradient_monitor(
                client_id=idx,
                local_weights=w,
                global_weights=global_weights_before
            )
            
            logger.info(f"  Client {idx}: Loss={loss:.4f}, Quality={quality_score:.3f}")
        
        # Aggregate with quality-weighted averaging
        aggregated_weights = gradfed_ppo.aggregate_with_quality_weights(
            client_weights=local_weights,
            selected_clients=selected_clients,
            data_sizes=data_sizes
        )
        
        # Update global model
        global_model.load_state_dict(aggregated_weights)
        
        # Calculate average training loss
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        # Evaluate global model
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        train_accuracy.append(test_acc)
        
        # Calculate reward for PPO
        comm_cost = len(selected_clients)
        reward = gradfed_ppo.calculate_reward_with_quality(
            prev_accuracy=prev_accuracy,
            curr_accuracy=test_acc,
            prev_loss=prev_loss,
            curr_loss=test_loss,
            selected_clients=selected_clients,
            comm_cost=comm_cost
        )
        
        # Construct next state
        next_base_state = [
            test_acc,
            test_loss,
            (epoch + 1) / args.epochs
        ]
        next_state = gradfed_ppo.get_enhanced_state(next_base_state, epoch + 1)
        next_state_tensor = torch.FloatTensor(next_state).to(device)
        
        # Store experience in PPO buffer
        action = selected_clients[0]  # Use first selected client as representative
        done = (epoch == args.epochs - 1)
        
        ppo.buffer.states.append(state_tensor)
        ppo.buffer.actions.append(torch.tensor(action))
        ppo.buffer.logprobs.append(torch.tensor(0.0))
        ppo.buffer.state_values.append(torch.tensor(0.0))
        ppo.buffer.rewards.append(reward)
        ppo.buffer.state_values.append(torch.tensor(0.0))
        ppo.buffer.is_terminals.append(done)
        
        # Update PPO policy every K rounds
        if False:  # Disabled PPO updates
            # logger.info("Updating PPO policy...")
            ppo.update()
        
        # Log round statistics
        gradfed_ppo.log_round_statistics(
            round_num=epoch + 1,
            selected_clients=selected_clients,
            accuracy=test_acc,
            loss=test_loss
        )
        
        # Print progress
        if (epoch + 1) % print_every == 0:
            print(f'\n{"="*70}')
            print(f'Round {epoch+1}/{args.epochs}:')
            print(f'  Avg Training Loss: {loss_avg:.4f}')
            print(f'  Test Accuracy: {100*test_acc:.2f}%')
            print(f'  Test Loss: {test_loss:.4f}')
            print(f'  Reward: {reward:.2f}')
            print(f'  Selected: {selected_clients}')
            print(f'  Quality: Avg={selection_info["avg_quality"]:.3f}, '
                  f'Noisy={selection_info["num_noisy"]}/{len(selected_clients)}')
            print(f'{"="*70}\n')
        
        # Update previous metrics
        prev_accuracy = test_acc
        prev_loss = test_loss
        
        # Save round statistics
        gradfed_ppo.round_accuracies.append(test_acc)
        gradfed_ppo.round_losses.append(test_loss)
    
    # Final evaluation
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    
    print(f'\n{"="*70}')
    print(f'Training Complete!')
    print(f'{"="*70}')
    print(f'Final Test Accuracy: {100*test_acc:.2f}%')
    print(f'Final Test Loss: {test_loss:.4f}')
    print(f'Training Time: {time.time()-start_time:.2f}s')
    print(f'{"="*70}')
    
    # Print gradient monitor statistics
    grad_stats = gradient_monitor.get_statistics()
    print(f'\n{"="*70}')
    print(f'Gradient Quality Statistics')
    print(f'{"="*70}')
    print(f'Average Quality: {grad_stats["avg_quality"]:.3f}')
    print(f'Quality Std Dev: {grad_stats.get("std_quality", 0):.3f}')
    print(f'Noisy Clients: {grad_stats["num_noisy"]}/{grad_stats["total_clients"]}')
    print(f'{"="*70}')
    
    # Print client participation
    print(f'\n{"="*70}')
    print(f'Client Participation')
    print(f'{"="*70}')
    participation_sorted = sorted(gradfed_ppo.client_participation.items(), 
                                 key=lambda x: x[1], reverse=True)
    print(f'Most active clients: {participation_sorted[:10]}')
    print(f'Least active clients: {participation_sorted[-10:]}')
    print(f'{"="*70}\n')
    
    # Save results
    save_dir = f'./results/fedppo_acc_record_{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save training history
    file_name = (f'gradfed_{args.dataset}_{args.model}_'
                f'users{args.num_users}_frac{args.frac}_'
                f'ep{args.epochs}_iid{args.iid}.npz')
    
    save_path = os.path.join(save_dir, file_name)
    gradfed_ppo.save_statistics(save_path)
    
    # Save accuracy and loss
    np.save(os.path.join(save_dir, f'gradfed_accuracy_{args.dataset}.npy'), 
            train_accuracy)
    np.save(os.path.join(save_dir, f'gradfed_loss_{args.dataset}.npy'), 
            train_loss)
    
    # Save model
    model_dir = './save'
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = f'{model_dir}/gradfed_{args.dataset}_{args.model}.pth'
    torch.save(global_model.state_dict(), model_save_path)
    logger.info(f'Model saved to {model_save_path}')
    
    print(f'\nResults saved to {save_dir}/')
    print(f'Model saved to {model_save_path}\n')


if __name__ == '__main__':
    gradfed_train()
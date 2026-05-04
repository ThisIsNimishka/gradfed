"""
Visualization and Analysis Script for GradFed Results
Place this in: src/visualize_results.py

Usage:
    python src/visualize_results.py --dataset cifar10 --methods gradfed fedppo favor
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 11


def load_results(dataset, method):
    """Load training results for a method"""
    base_dir = f'./results/fedppo_acc_record_{dataset}'
    
    results = {}
    
    # Load accuracy
    acc_file = f'{base_dir}/{method}_accuracy_{dataset}.npy'
    if os.path.exists(acc_file):
        results['accuracy'] = np.load(acc_file)
    
    # Load loss
    loss_file = f'{base_dir}/{method}_loss_{dataset}.npy'
    if os.path.exists(loss_file):
        results['loss'] = np.load(loss_file)
    
    # Load detailed stats (for GradFed)
    stats_pattern = f'{base_dir}/{method}_{dataset}_*.npz'
    import glob
    stats_files = glob.glob(stats_pattern)
    
    if stats_files:
        stats = np.load(stats_files[0], allow_pickle=True)
        results['stats'] = {key: stats[key] for key in stats.files}
    
    return results


def plot_accuracy_comparison(results_dict, dataset, save_dir):
    """Plot accuracy comparison across methods"""
    plt.figure(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (method, results) in enumerate(results_dict.items()):
        if 'accuracy' in results:
            acc = results['accuracy']
            rounds = range(1, len(acc) + 1)
            
            plt.plot(rounds, acc * 100, label=method.upper(), 
                    linewidth=2.5, color=colors[idx % len(colors)],
                    marker='o', markersize=3, markevery=max(1, len(acc)//20))
    
    plt.xlabel('Communication Rounds', fontsize=13, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    plt.title(f'Test Accuracy Comparison - {dataset.upper()}', 
             fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = f'{save_dir}/accuracy_comparison_{dataset}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_loss_comparison(results_dict, dataset, save_dir):
    """Plot loss comparison across methods"""
    plt.figure(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (method, results) in enumerate(results_dict.items()):
        if 'loss' in results:
            loss = results['loss']
            rounds = range(1, len(loss) + 1)
            
            plt.plot(rounds, loss, label=method.upper(), 
                    linewidth=2.5, color=colors[idx % len(colors)],
                    marker='s', markersize=3, markevery=max(1, len(loss)//20))
    
    plt.xlabel('Communication Rounds', fontsize=13, fontweight='bold')
    plt.ylabel('Test Loss', fontsize=13, fontweight='bold')
    plt.title(f'Test Loss Comparison - {dataset.upper()}', 
             fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = f'{save_dir}/loss_comparison_{dataset}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_gradient_quality_heatmap(results, save_dir):
    """Plot gradient quality scores as heatmap (GradFed only)"""
    if 'stats' not in results or 'quality_scores' not in results['stats']:
        return
    
    quality_scores = results['stats']['quality_scores'].item()
    
    if not quality_scores:
        return
    
    # Prepare data
    clients = sorted(quality_scores.keys())
    scores = [quality_scores[c] for c in clients]
    
    # Create heatmap data (reshape for visualization)
    n_clients = len(clients)
    n_cols = min(10, n_clients)
    n_rows = (n_clients + n_cols - 1) // n_cols
    
    heatmap_data = np.zeros((n_rows, n_cols))
    for idx, score in enumerate(scores):
        row = idx // n_cols
        col = idx % n_cols
        heatmap_data[row, col] = score
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1, cbar_kws={'label': 'Quality Score'},
                linewidths=0.5, linecolor='gray')
    
    plt.title('Client Gradient Quality Scores (GradFed)', 
             fontsize=15, fontweight='bold')
    plt.xlabel('Client ID (column)', fontsize=12)
    plt.ylabel('Client ID (row)', fontsize=12)
    plt.tight_layout()
    
    save_path = f'{save_dir}/gradient_quality_heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_client_participation(results, save_dir):
    """Plot client participation distribution (GradFed only)"""
    if 'stats' not in results or 'client_participation' not in results['stats']:
        return
    
    participation = results['stats']['client_participation'].item()
    
    if not participation:
        return
    
    clients = sorted(participation.keys())
    counts = [participation[c] for c in clients]
    
    plt.figure(figsize=(12, 6))
    
    # Bar plot
    bars = plt.bar(clients, counts, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Color bars by quality if available
    if 'quality_scores' in results['stats']:
        quality_scores = results['stats']['quality_scores'].item()
        for client, bar in zip(clients, bars):
            if client in quality_scores:
                score = quality_scores[client]
                if score < 0.3:
                    bar.set_color('red')
                    bar.set_alpha(0.8)
                elif score < 0.5:
                    bar.set_color('orange')
                    bar.set_alpha(0.7)
    
    plt.xlabel('Client ID', fontsize=13, fontweight='bold')
    plt.ylabel('Participation Count', fontsize=13, fontweight='bold')
    plt.title('Client Participation Distribution', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    save_path = f'{save_dir}/client_participation.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_convergence_speed(results_dict, dataset, save_dir, target_acc=0.6):
    """Plot rounds to reach target accuracy"""
    methods = []
    rounds_to_target = []
    
    for method, results in results_dict.items():
        if 'accuracy' not in results:
            continue
        
        acc = results['accuracy']
        
        # Find first round reaching target
        for round_num, accuracy in enumerate(acc, 1):
            if accuracy >= target_acc:
                methods.append(method.upper())
                rounds_to_target.append(round_num)
                break
    
    if not methods:
        print(f"No method reached {target_acc*100}% accuracy")
        return
    
    plt.figure(figsize=(10, 6))
    
    colors = ['#2ca02c' if 'GRADFED' in m else '#1f77b4' for m in methods]
    bars = plt.bar(methods, rounds_to_target, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Rounds to Target', fontsize=13, fontweight='bold')
    plt.title(f'Convergence Speed (Target: {target_acc*100}% Accuracy)', 
             fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    save_path = f'{save_dir}/convergence_speed_{dataset}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def create_summary_table(results_dict, dataset, save_dir):
    """Create summary statistics table"""
    summary = []
    
    for method, results in results_dict.items():
        if 'accuracy' not in results:
            continue
        
        acc = results['accuracy']
        loss = results.get('loss', [])
        
        row = {
            'Method': method.upper(),
            'Final Acc (%)': f"{acc[-1]*100:.2f}",
            'Best Acc (%)': f"{np.max(acc)*100:.2f}",
            'Avg Last 10 (%)': f"{np.mean(acc[-10:])*100:.2f}",
        }
        
        if len(loss) > 0:
            row['Final Loss'] = f"{loss[-1]:.4f}"
            row['Best Loss'] = f"{np.min(loss):.4f}"
        
        # GradFed specific
        if 'stats' in results and 'gradient_stats' in results['stats']:
            grad_stats = results['stats']['gradient_stats'].item()
            row['Avg Quality'] = f"{grad_stats.get('avg_quality', 0):.3f}"
            row['Noisy Clients'] = f"{grad_stats.get('num_noisy', 0)}/{grad_stats.get('total_clients', 0)}"
        
        summary.append(row)
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(14, len(summary) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Convert to list of lists
    headers = list(summary[0].keys())
    data = [[row[h] for h in headers] for row in summary]
    
    table = ax.table(cellText=data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    colWidths=[0.15] * len(headers))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    # Highlight GradFed row
    for i, row in enumerate(summary, 1):
        if 'GRADFED' in row['Method']:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#C5E0B4')
    
    plt.title('Performance Summary', fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    
    save_path = f'{save_dir}/summary_table_{dataset}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    # Also save as text
    text_path = f'{save_dir}/summary_{dataset}.txt'
    with open(text_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Performance Summary - {dataset.upper()}\n")
        f.write("=" * 80 + "\n\n")
        
        for row in summary:
            f.write(f"{row['Method']}:\n")
            for key, value in row.items():
                if key != 'Method':
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print(f"Saved: {text_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize GradFed Results')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='Dataset name (cifar10, cifar100, mnist)')
    parser.add_argument('--methods', nargs='+', 
                       default=['gradfed', 'fedppo', 'favor', 'fedavg'],
                       help='Methods to compare')
    parser.add_argument('--save_dir', type=str, default='./visualization_results',
                       help='Directory to save plots')
    parser.add_argument('--target_acc', type=float, default=0.6,
                       help='Target accuracy for convergence speed')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Visualizing Results for {args.dataset.upper()}")
    print(f"{'='*80}\n")
    
    # Load results
    results_dict = {}
    for method in args.methods:
        print(f"Loading {method}...")
        results = load_results(args.dataset, method)
        if results:
            results_dict[method] = results
            print(f"  ✓ Loaded {list(results.keys())}")
        else:
            print(f"  ✗ No results found")
    
    if not results_dict:
        print("\nNo results found! Please run experiments first.")
        return
    
    print(f"\n{'='*80}")
    print("Generating Visualizations")
    print(f"{'='*80}\n")
    
    # Generate plots
    plot_accuracy_comparison(results_dict, args.dataset, args.save_dir)
    plot_loss_comparison(results_dict, args.dataset, args.save_dir)
    plot_convergence_speed(results_dict, args.dataset, args.save_dir, args.target_acc)
    create_summary_table(results_dict, args.dataset, args.save_dir)
    
    # GradFed specific plots
    if 'gradfed' in results_dict:
        print("\nGenerating GradFed-specific plots...")
        plot_gradient_quality_heatmap(results_dict['gradfed'], args.save_dir)
        plot_client_participation(results_dict['gradfed'], args.save_dir)
    
    print(f"\n{'='*80}")
    print(f"All visualizations saved to: {args.save_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
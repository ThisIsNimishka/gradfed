#!/bin/bash

# Comparison script to evaluate GradFed against baseline methods
# Usage: ./scripts/run_comparison.sh (from repo root or scripts directory)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==================================================================="
echo "GradFed vs Baselines Benchmark Suite"
echo "Repository Root: $REPO_ROOT"
echo "==================================================================="

# Configuration
DATASET="cifar10"
MODEL="resnet10"
NUM_USERS=100
FRAC=0.1
LOCAL_EP=10
LOCAL_BS=10
EPOCHS=100
GPU=0
IID=0  # 0 for non-IID, 1 for IID

# Semantic results and logs directory
RESULTS_DIR="$REPO_ROOT/results/comparison"
LOGS_DIR="$REPO_ROOT/logs"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Model: $MODEL"
echo "  Num Users: $NUM_USERS"
echo "  Fraction: $FRAC"
echo "  Local Epochs: $LOCAL_EP"
echo "  Global Rounds: $EPOCHS"
echo "  Non-IID: $(if [ $IID -eq 0 ]; then echo 'Yes'; else echo 'No'; fi)"
echo "  Results Directory: $RESULTS_DIR"
echo ""

# Function to run experiment
run_experiment() {
    local method=$1
    local script_rel=$2
    local logfile="${LOGS_DIR}/${method}_${DATASET}.log"
    
    echo "-------------------------------------------------------------------"
    echo "Running: $method ($script_rel)"
    echo "Log file: $logfile"
    echo "-------------------------------------------------------------------"
    
    python "$REPO_ROOT/$script_rel" \
        --dataset $DATASET \
        --model $MODEL \
        --num_users $NUM_USERS \
        --frac $FRAC \
        --local_ep $LOCAL_EP \
        --local_bs $LOCAL_BS \
        --epochs $EPOCHS \
        --gpu $GPU \
        --iid $IID \
        2>&1 | tee "$logfile"
    
    echo ""
    echo "$method completed!"
    echo ""
}

# 1. Run FedAvg baseline
echo "==================================================================="
echo "1. Running FedAvg (Baseline)"
echo "==================================================================="
run_experiment "FedAvg" "baseline/fedavg_main.py"

# 2. Run FedProx baseline
echo "==================================================================="
echo "2. Running FedProx (Baseline)"
echo "==================================================================="
run_experiment "FedProx" "baseline/fedprox_main.py"

# 3. Run FAVOR baseline (if ddqn dependencies present)
if [ -d "$REPO_ROOT/baseline/ddqn_server" ] || [ -f "$REPO_ROOT/baseline/ddqn_server/DDQN.py" ]; then
    echo "==================================================================="
    echo "3. Running FAVOR (RL Baseline)"
    echo "==================================================================="
    run_experiment "FAVOR" "baseline/favor_main.py"
else
    echo "Skipping FAVOR baseline (baseline/ddqn_server submodule not present)."
fi

# 4. Run original FedPPO
echo "==================================================================="
echo "4. Running FedPPO (Original)"
echo "==================================================================="
run_experiment "FedPPO" "src/ppo_remove_main.py"

# 5. Run GradFed (Proposed Method)
echo "==================================================================="
echo "5. Running GradFed (Proposed Method)"
echo "==================================================================="
run_experiment "GradFed" "src/gradfed_main.py"

echo "==================================================================="
echo "All experiments completed!"
echo "==================================================================="

# Generate comparison report
python - << EOF
import numpy as np
import matplotlib.pyplot as plt
import os

repo_root = "$REPO_ROOT"
results_dir = "$RESULTS_DIR"
dataset = "$DATASET"

methods = ['FedAvg', 'FedProx', 'FAVOR', 'FedPPO', 'GradFed']
colors = ['blue', 'green', 'orange', 'red', 'purple']

# Helper to find file in results/ or fedppo_acc_record_
def find_metric_file(method_name, metric_type):
    candidates = [
        os.path.join(repo_root, 'results', dataset, f'{method_name.lower()}_{metric_type}_{dataset}.npy'),
        os.path.join(repo_root, f'fedppo_acc_record_{dataset}', f'{method_name.lower()}_{metric_type}_{dataset}.npy')
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

# Load accuracy results
plt.figure(figsize=(12, 5))

# Plot 1: Accuracy over rounds
plt.subplot(1, 2, 1)
for method, color in zip(methods, colors):
    try:
        acc_file = find_metric_file(method, 'accuracy')
        if acc_file:
            accuracy = np.load(acc_file)
            plt.plot(accuracy, label=method, color=color, linewidth=2)
    except Exception as e:
        print(f"Could not load {method}: {e}")

plt.xlabel('Communication Rounds', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Loss over rounds
plt.subplot(1, 2, 2)
for method, color in zip(methods, colors):
    try:
        loss_file = find_metric_file(method, 'loss')
        if loss_file:
            loss = np.load(loss_file)
            plt.plot(loss, label=method, color=color, linewidth=2)
    except Exception as e:
        pass

plt.xlabel('Communication Rounds', fontsize=12)
plt.ylabel('Test Loss', fontsize=12)
plt.title('Test Loss Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
save_plot = os.path.join(results_dir, f'comparison_plot_{dataset}.png')
plt.savefig(save_plot, dpi=300)
print(f"\nComparison plot saved to: {save_plot}")

# Print summary statistics
print("\n=================================================================")
print("Summary Statistics")
print("=================================================================")
for method in methods:
    acc_file = find_metric_file(method, 'accuracy')
    if acc_file:
        try:
            accuracy = np.load(acc_file)
            print(f"\n{method}:")
            print(f"  Final Accuracy: {accuracy[-1]:.4f}")
            print(f"  Best Accuracy: {np.max(accuracy):.4f}")
            print(f"  Average (last 10 rounds): {np.mean(accuracy[-10:]):.4f}")
        except Exception:
            print(f"\n{method}: Error loading accuracy file")
    else:
        print(f"\n{method}: Results not found")

print("\n=================================================================")
EOF

echo ""
echo "Comparison report generated!"
echo "Check $RESULTS_DIR/ for summary plots and $LOGS_DIR/ for logs."
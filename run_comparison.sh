#!/bin/bash

# Comparison script to evaluate GradFed against baseline methods
# Place this file in the root directory: run_comparison.sh
# Make executable: chmod +x run_comparison.sh

echo "==================================================================="
echo "GradFed vs Baselines Comparison"
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

# Create results directory
RESULTS_DIR="./comparison_results"
mkdir -p $RESULTS_DIR

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Model: $MODEL"
echo "  Num Users: $NUM_USERS"
echo "  Fraction: $FRAC"
echo "  Local Epochs: $LOCAL_EP"
echo "  Global Rounds: $EPOCHS"
echo "  Non-IID: $(if [ $IID -eq 0 ]; then echo 'Yes'; else echo 'No'; fi)"
echo ""

# Function to run experiment
run_experiment() {
    local method=$1
    local script=$2
    local logfile="${RESULTS_DIR}/${method}_${DATASET}.log"
    
    echo "-------------------------------------------------------------------"
    echo "Running: $method"
    echo "Log file: $logfile"
    echo "-------------------------------------------------------------------"
    
    python $script \
        --dataset $DATASET \
        --model $MODEL \
        --num_users $NUM_USERS \
        --frac $FRAC \
        --local_ep $LOCAL_EP \
        --local_bs $LOCAL_BS \
        --epochs $EPOCHS \
        --gpu $GPU \
        --iid $IID \
        2>&1 | tee $logfile
    
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

# 3. Run FAVOR baseline
echo "==================================================================="
echo "3. Running FAVOR (RL Baseline)"
echo "==================================================================="
run_experiment "FAVOR" "baseline/favor_main.py"

# 4. Run original FedPPO
echo "==================================================================="
echo "4. Running FedPPO (Original)"
echo "==================================================================="
run_experiment "FedPPO" "src/ppo_remove_main.py"

# 5. Run GradFed (Your method)
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

results_dir = "$RESULTS_DIR"
dataset = "$DATASET"

methods = ['FedAvg', 'FedProx', 'FAVOR', 'FedPPO', 'GradFed']
colors = ['blue', 'green', 'orange', 'red', 'purple']

# Load accuracy results
plt.figure(figsize=(12, 5))

# Plot 1: Accuracy over rounds
plt.subplot(1, 2, 1)
for method, color in zip(methods, colors):
    try:
        acc_file = f'./fedppo_acc_record_{dataset}/{method.lower()}_accuracy_{dataset}.npy'
        if os.path.exists(acc_file):
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
        loss_file = f'./fedppo_acc_record_{dataset}/{method.lower()}_loss_{dataset}.npy'
        if os.path.exists(loss_file):
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
plt.savefig(f'{results_dir}/comparison_plot_{dataset}.png', dpi=300)
print(f"\\nComparison plot saved to: {results_dir}/comparison_plot_{dataset}.png")

# Print summary statistics
print("\\n=================================================================")
print("Summary Statistics")
print("=================================================================")
for method in methods:
    try:
        acc_file = f'./fedppo_acc_record_{dataset}/{method.lower()}_accuracy_{dataset}.npy'
        if os.path.exists(acc_file):
            accuracy = np.load(acc_file)
            print(f"\\n{method}:")
            print(f"  Final Accuracy: {accuracy[-1]:.4f}")
            print(f"  Best Accuracy: {np.max(accuracy):.4f}")
            print(f"  Average (last 10 rounds): {np.mean(accuracy[-10:]):.4f}")
    except:
        print(f"\\n{method}: Results not found")

print("\\n=================================================================")
EOF

echo ""
echo "Comparison report generated!"
echo "Check $RESULTS_DIR/ for logs and plots"
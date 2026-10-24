# GradFed Setup & User Guide

## Overview
**GradFed (Gradient-Aware Federated Learning)** is an intelligent client selection and robust aggregation framework designed for non-IID and noisy federated learning environments.

## Repository Layout

```
gradfed/
├── src/                              # Core algorithmic implementation
│   ├── gradient_monitor.py           # Gradient quality, stability & cosine similarity tracking
│   ├── gradfed_main.py               # Main GradFed training pipeline
│   ├── models.py                     # ResNet, CNN, and MLP model architectures
│   ├── options.py                    # Hyperparameter & CLI parsing
│   ├── sampling.py                   # Dirichlet Non-IID and IID dataset partitioners
│   ├── update.py                     # Local updates and validation inference
│   ├── utils.py                      # Data loaders and dataset preprocessors
│   ├── visualize_results.py          # Results analysis and plotting suite
│   └── ppo_server/
│       ├── gradfed_ppo_update.py     # Quality-aware client selection & weighted aggregation
│       ├── ppo.py                    # Actor-Critic PPO implementation
│       └── ppo_remove_update.py      # Upstream FedPPO clustering baseline
├── baseline/                         # Classical FL algorithms (FedAvg, FedProx, etc.)
│   ├── fedavg_main.py
│   ├── fedprox_main.py
│   ├── favor_main.py
│   └── kmeans_main.py
├── data/                             # Dataset cache folder
│   └── README.md
├── docs/                             # Documentation and guides
│   ├── setup_guide.md                # This setup guide
│   └── assets/                       # Vector SVGs and diagrams
├── environments/                     # Environment specifications & legacy locks
│   ├── legacy_fedppo_env.yaml
│   └── requirements_gradfed.txt
├── logs/                             # Execution and benchmark logs
│   ├── gradfed_training.log
│   └── experiment_30rounds.log
├── results/                          # Evaluation outputs, metrics, and checkpoints
│   ├── cifar10/                      # Saved accuracy/loss arrays and run stats
│   └── checkpoints/                  # Saved PyTorch models (.pth) and baseline pickles
├── scripts/                          # Automated execution and validation scripts
│   ├── run_comparison.sh             # End-to-end benchmark comparison script
│   └── setup_verification.py         # System verification and diagnostics script
├── app.js                            # Interactive web simulator logic
├── index.css                         # Showcase stylesheet
├── index.html                        # Live interactive simulator web app
├── vercel.json                       # Vercel deployment configuration
├── requirements.txt                  # Primary Python pip dependencies
└── environment.yml                   # Primary cross-platform Conda environment
```

## Installation Steps

### Step 1: Set Up Python Environment

Using **Conda** (recommended):
```bash
conda env create -f environment.yml
conda activate gradfed
```

Or using **Pip**:
```bash
pip install -r requirements.txt
```

### Step 2: Verify Setup

Run the built-in diagnostic tool to verify environment and components:
```bash
python scripts/setup_verification.py
```

### Step 3: Test GradFed

Run a quick smoke-test with a small configuration:

```bash
python src/gradfed_main.py \
    --dataset cifar10 \
    --model resnet10 \
    --num_users 20 \
    --frac 0.2 \
    --local_ep 5 \
    --local_bs 10 \
    --epochs 10 \
    --gpu 0 \
    --iid 0
```

Expected output:
```
GradientMonitor initialized
PPO initialized - State dim: 6, Action dim: 20
Selected 4 clients
Client 0: Loss=2.3045, Quality=0.876
...
Round 10/10:
  Avg Training Loss: 1.8234
  Test Accuracy: 35.67%
  Quality: Avg=0.654, Noisy=1/4
```

## Running Experiments

### Quick Test (10 rounds)
```bash
python src/gradfed_main.py --dataset cifar10 --model resnet10 \
    --num_users 50 --frac 0.1 --local_ep 5 --epochs 10 --gpu 0
```

### Full Benchmark Experiment (30–100 rounds)
```bash
python src/gradfed_main.py --dataset cifar10 --model resnet10 \
    --num_users 50 --frac 0.2 --local_ep 10 --epochs 30 --gpu 0
```

### Compare with Baselines
```bash
# Execute automated baseline benchmark suite
chmod +x scripts/run_comparison.sh
./scripts/run_comparison.sh
```

## Key Parameters

| Parameter | Description | Recommended Values |
|---|---|:---:|
| `--num_users` | Total clients | `50-100` |
| `--frac` | Fraction selected per round | `0.1-0.2` |
| `--local_ep` | Local epochs | `5-10` |
| `--local_bs` | Local batch size | `10-50` |
| `--epochs` | Global rounds | `30-100` |
| `--iid` | Data distribution | `0` (non-IID), `1` (IID) |
| `--gpu` | GPU device | `0`, `1`, or `-1` (CPU) |

## Understanding the Output

### During Training

```
=== Round 15 Statistics ===
Accuracy: 0.6234, Loss: 1.1234
Selected clients: [12, 45, 17, 23, 39, 34, 46, 28, 40, 3]
Gradient Quality - Avg: 0.687, Noisy: 2/10
  Client 12: Quality=0.876, Participations=5
  Client 45: Quality=0.234, Participations=12  [LOW QUALITY - NOISY]
```

**Quality Score Interpretation:**
- **0.7–1.0**: High quality, stable gradients
- **0.4–0.7**: Medium quality, acceptable
- **0.0–0.4**: Low quality, potentially noisy (marked as NOISY)

### Aggregation Weights

```
Aggregation weights - 
  Data: [0.15, 0.20, 0.10, ...]
  Quality: [0.12, 0.18, 0.08, ...]
  Final: [0.14, 0.19, 0.09, ...]
```

Shows how data size and gradient quality combine to weight client contributions.

### Reward Components

```
Reward components - 
  Acc: 2.34        # Accuracy gain
  Loss: 1.23       # Loss reduction
  Quality: 3.45    # Quality bonus
  Diversity: 0.89  # Diversity bonus
  Comm: -1.00      # Communication penalty
  Total: 6.91
```

## Monitoring Training

### View Live Logs
```bash
tail -f logs/gradfed_training.log
```

### Check Results
```bash
# View saved statistics
python -c "
import numpy as np
data = np.load('./results/cifar10/gradfed_cifar10_resnet10_users50_frac0.2_ep30_iid0.npz')
print('Keys:', list(data.keys()))
print('Final accuracy:', data['round_accuracies'][-1])
print('Quality scores:', data['quality_scores'])
"
```

### Plot Results
```python
import numpy as np
import matplotlib.pyplot as plt

# Load accuracy
acc = np.load('./results/cifar10/gradfed_accuracy_cifar10.npy')

plt.figure(figsize=(10, 6))
plt.plot(acc, linewidth=2, color='#6366f1')
plt.xlabel('Round')
plt.ylabel('Test Accuracy')
plt.title('GradFed Training Progress')
plt.grid(True, alpha=0.3)
plt.savefig('./results/gradfed_progress.png')
```

## Troubleshooting

### Issue 1: Import Errors
```
ModuleNotFoundError: No module named 'gradient_monitor'
```
**Solution:** Ensure you are running from the repository root, or that `src/` is in your `PYTHONPATH`.

### Issue 2: Model Loading Errors
```
Error: unrecognized model
```
**Solution:** Check that your `models.py` has the required model class (e.g. `resnet10`, `cnn`, `mlp`).

### Issue 3: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size or number of selected clients:
```bash
python src/gradfed_main.py --local_bs 5 --frac 0.1 --gpu 0
```

## Expected Performance (Non-IID CIFAR-10)

| Method | Top-1 Accuracy | Rounds to 60% Acc | Quality Aware? |
|:---|:---:|:---:|:---:|
| FedAvg | 62.3% | 85 | ❌ |
| FedPPO | 67.8% | 70 | ⚠️ |
| **GradFed** | **71.2%** | **60** | ✅ |

**GradFed advantages:**
- ✅ **+3.4%** accuracy improvement over FedPPO (+8.9% over FedAvg)
- ✅ **15–25%** faster convergence speed
- ✅ Robust filtering of low-quality/noisy clients
- ✅ Dual-balanced quality and data volume aggregation

## Next Steps

1. **Run baseline comparison**: `chmod +x scripts/run_comparison.sh && ./scripts/run_comparison.sh`
2. **Analyze gradient quality**: Check which clients are consistently noisy
3. **Tune hyperparameters**: Adjust quality threshold (`--quality_threshold`), window size (`--window_size`)
4. **Test with noise**: Add label noise to validate robustness

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{jain2025gradfed,
  title={GradFed: Gradient-Aware Client Selection in Federated Learning},
  author={Jain, Nimishka},
  year={2025},
  school={National Institute of Technology Goa},
  address={Goa, India},
  supervisor={Swain, Pravati}
}
```

## Support

For issues or questions:
1. Check logs: [logs/gradfed_training.log](file:///logs/gradfed_training.log)
2. Review comparison results: `results/comparison/`
3. Run verification diagnostics: `python scripts/setup_verification.py`
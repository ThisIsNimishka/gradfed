# GradFed

GradFed is a high-fidelity research infrastructure for federated learning experimentation with noisy-client robustness and policy-enabled scheduling. The repository operationalizes hierarchical/statistical clustering, PPO-centric policy optimization, and gradient aggregation for both vision and sequential modalities.

## 🚀 Execution environment

Bootstrap the runtime environment using the supplied Conda manifest:

    conda env create -f fedppo_env.yaml

Alternatively, resolve the dependency graph directly:

    pip install -r requirements_gradfed.txt

## 📦 Supported corpus modalities

- `cifar10`
- `cifar100`
- `emnist`
- `mnist`
- `fmnist`
- `imdb`
- `clothing1m` (requires local dataset artifacts)

## 🔧 Architectural modules

- `src/gradfed_main.py` — primary federated orchestration with model aggregation, gradient averaging, and heterogeneity management
- `src/ppo_remove_main.py` — PPO-driven noisy-client detection and adaptive scheduling
- `src/visualize_results.py` — post hoc metric extraction and comparative convergence visualization
- `src/options.py` — centralized CLI schema for dataset semantics, optimization hyperparameters, and noise injection controls
- `src/utils.py` — dataset ingestion, user partitioning, noising primitives, and federated aggregation utilities
- `baselines/` — baseline federated learning implementations (FedAvg, FedProx, etc.)
- `scripts/` — utility scripts for experiment comparison and setup verification
- `docs/` — documentation and setup guides
- `results/` — experiment outputs, model checkpoints, and performance records
- `data/` — dataset storage and metadata

## ⚙️ Parameter lexicon

| Parameter | Flag | Default | Semantics |
|---|---|---|---|
| Data distribution | `--iid` | `1` | `1`: IID partitioning; `0`: Dirichlet-driven non-IID heterogeneity |
| Dataset selector | `--dataset` | `cifar10` | Supported tokens: `cifar10`, `cifar100`, `emnist`, `mnist`, `fmnist`, `imdb`, `clothing1m` |
| Global sync rounds | `--epochs` | `10` | Number of global federated communication rounds |
| Client sampling ratio | `--frac` | `0.2` | Fraction of clients sampled per round |
| Local update epochs | `--local_ep` | `10` | Iterations per local update |
| Local minibatch size | `--local_bs` | `10` | Mini-batch cardinality for local SGD |
| Learning rate | `--lr` | `0.01` | Base optimizer learning rate |
| Model topology | `--model` | `mlp` | e.g. `resnet10`, `resnet18`, `resnet50`, `lstm`, `cnn`, `dnn` |
| Compute binding | `--gpu_id` | `None` | CUDA device index; omit to execute on CPU |
| Non-IID class probability | `--non_iid_prob_class` | `0.7` | Class bias control for Dirichlet partitioning |
| Dirichlet concentration | `--alpha_dirichlet` | `10` | Controls non-IID severity |
| Noisy client ratio | `--level_n_system` | `0.0` | Fraction of clients with label corruption |
| Noise lower bound | `--level_n_lowerb` | `0.0` | Minimum noise intensity |
| Noise distribution | `--noise_type` | `symmetric` | Label corruption schema |
| Reward threshold | `--target_acc` | `0.8` | PPO reward shaping threshold |
| Clustering backend | `--cluster_method` | `hier` | `hier`, `gmm`, `dbscan`, `kmeans`, `no` |

## 🧪 Usage patterns

### Federated training pipeline

    python src/gradfed_main.py --dataset cifar10 --model resnet10 --num_users 50 \
      --frac 0.2 --local_ep 10 --local_bs 10 --epochs 100 --gpu_id 0

### PPO-driven noisy-client scheduling

    python src/ppo_remove_main.py --dataset cifar10 --model resnet10 --iid 0 \
      --epochs 100 --level_n_system 0.0 --level_n_lowerb 0.0 --target_acc 0.9 \
      --frac 0.2 --non_iid_prob_class 0.7 --alpha_dirichlet 10 --gpu_id 0

### Comparative analysis

    python src/visualize_results.py --dataset cifar10 --methods gradfed \
      --save_dir ./visualization_results --target_acc 0.6

## 📝 Implementation caveats

- `cifar10`, `cifar100`, and `emnist` are automatically retrieved if missing.
- `imdb` ingestion leverages `keras.datasets.imdb` with fixed-length sequence padding.
- `clothing1m` requires pre-provisioned local dataset artifacts and is not bootstrapped automatically.

## 📚 Citation
If you leverage this repository in research, please cite:

    @ARTICLE{10909702,
      author={Zheyu Zhao and Anran Li and Ruidong Li and Lei Yang and Xiaohua Xu},
      journal={IEEE Transactions on Cognitive Communications and Networking},
      title={FedPPO: Reinforcement Learning-Based Client Selection for Federated Learning With Heterogeneous Data},
      year={2025},
      pages={1-1},
      doi={10.1109/TCCN.2025.3547751}

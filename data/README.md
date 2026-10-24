# Datasets Directory

This directory stores raw and processed benchmark datasets used by **GradFed**.

## Supported Datasets

| Dataset | Type | Classes | Default Model | Automatic Download? |
| :--- | :--- | :---: | :---: | :---: |
| **CIFAR-10** | Image classification (32×32 RGB) | 10 | ResNet-10 / CNN | ✅ Yes (`torchvision.datasets.CIFAR10`) |
| **CIFAR-100** | Fine-grained image classification | 100 | ResNet-10 / ResNet-18 | ✅ Yes (`torchvision.datasets.CIFAR100`) |
| **MNIST** | Handwritten digits (28×28 grayscale) | 10 | CNN / MLP | ✅ Yes (`torchvision.datasets.MNIST`) |
| **Fashion-MNIST** | Clothing articles (28×28 grayscale) | 10 | CNN / MLP | ✅ Yes (`torchvision.datasets.FashionMNIST`) |
| **EMNIST** | Extended handwritten characters | 62 (`byclass`) | ResNet-10 / CNN | ✅ Yes (`torchvision.datasets.EMNIST`) |
| **IMDB** | Binary text sentiment classification | 2 | LSTM / DNN | ✅ Yes (`keras.datasets.imdb`) |

## Data Partitioning Schemes

GradFed partitions client data in `src/sampling.py` using two primary methods:

1. **Non-IID Dirichlet Partitioning (`--iid 0`)**:
   - Uses a Dirichlet distribution with concentration parameter $\alpha$ (`--alpha_dirichlet 10.0`).
   - Lower values ($\alpha < 1.0$) model severe class skew across clients.
   - Higher values ($\alpha \ge 10.0$) model realistic heterogeneous edge partitions.
2. **IID Partitioning (`--iid 1`)**:
   - Uniform random sampling across all client nodes.

## Directory Structure
Downloaded files will be cached under subdirectories:
```
data/
├── cifar10/
├── cifar100/
├── mnist/
├── fmnist/
└── emnist/
```
*(Note: Large binary dataset caches are automatically ignored by `.gitignore` and `.vercelignore` to avoid cluttering version control).*

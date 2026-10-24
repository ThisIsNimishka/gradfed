#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standard Local Update and Evaluation for Federated Learning Baselines
"""

import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """Dataset wrapper for subset of dataset indices."""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.as_tensor(image), torch.as_tensor(label)


class LocalUpdate(object):
    """Local client training for FedAvg and baseline algorithms."""
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.dataset_len = len(idxs)
        self.trainloader = DataLoader(
            DatasetSplit(dataset, idxs),
            batch_size=self.args.local_bs,
            shuffle=True
        )
        gpu_id = getattr(args, 'gpu_id', None)
        if gpu_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_id}')
        else:
            self.device = torch.device('cpu')
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []

        if getattr(self.args, 'optimizer', 'sgd') == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.args.lr,
                momentum=getattr(self.args, 'momentum', 0.5)
            )
        elif getattr(self.args, 'optimizer', 'sgd') == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.args.lr,
                weight_decay=1e-4
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.args.lr,
                momentum=0.5
            )

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

            epoch_loss.append(sum(batch_loss) / max(len(batch_loss), 1))

        avg_loss = sum(epoch_loss) / max(len(epoch_loss), 1)
        return model, model.state_dict(), avg_loss


class FedProxLocalUpdate(LocalUpdate):
    """Local client training with proximal term for FedProx."""
    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []
        global_weight_tensors = [p.data.clone() for p in model.parameters()]
        mu = getattr(self.args, 'mu', 0.01)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.args.lr,
            momentum=getattr(self.args, 'momentum', 0.5)
        )

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)

                # Proximal term: (mu / 2) * ||w - w_t||^2
                if mu > 0:
                    prox_loss = 0.0
                    for w, w_t in zip(model.parameters(), global_weight_tensors):
                        prox_loss += torch.sum((w - w_t) ** 2)
                    loss = loss + (mu / 2.0) * prox_loss

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / max(len(batch_loss), 1))

        avg_loss = sum(epoch_loss) / max(len(epoch_loss), 1)
        return model, model.state_dict(), avg_loss


def test_inference(args, model, test_dataset):
    """Evaluate global model test accuracy and loss."""
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    gpu_id = getattr(args, 'gpu_id', None)
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')

    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    accuracy = correct / max(total, 1)
    avg_loss = loss / max(len(testloader), 1)
    return accuracy, avg_loss

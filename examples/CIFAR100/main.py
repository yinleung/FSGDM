#!/usr/bin/env python
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms

from tqdm import tqdm

from resnet import resnet18, resnet34, resnet50, resnet101, resnet152

from fsgdm import FSGDM

def get_data_loaders(data_dir, batch_size):
    """
    Returns the training and test dataloaders for CIFAR-100.
    """
    # CIFAR-100 normalization values (computed from the training set)
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    # Define transforms for training and testing
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),                
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Download (if needed) and load the CIFAR-100 dataset
    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transforms)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


def build_model(arch='resnet50'):
    """
    Build and return a ResNet model for CIFAR-100.

    Args:
        arch (str): The ResNet architecture to build. Options: 'resnet18', 'resnet34',
                    'resnet50', 'resnet101', 'resnet152'.

    Returns:
        nn.Module: The constructed ResNet model.
    """
    net_opt = {}  # Options to pass to ResNet, like number of blocks and output classes
    net_opt['num_classes'] = 100  # CIFAR-100 has 100 classes
    
    if arch == 'resnet18':
        net_opt['num_block'] = [2, 2, 2, 2]
        model = resnet18(net_opt)
    elif arch == 'resnet34':
        net_opt['num_block'] = [3, 4, 6, 3]
        model = resnet34(net_opt)
    elif arch == 'resnet50':
        net_opt['num_block'] = [3, 4, 6, 3]
        model = resnet50(net_opt)
    elif arch == 'resnet101':
        net_opt['num_block'] = [3, 4, 23, 3]
        model = resnet101(net_opt)
    elif arch == 'resnet152':
        net_opt['num_block'] = [3, 8, 36, 3]
        model = resnet152(net_opt)
    else:
        raise ValueError("Unsupported architecture: {}".format(arch))
    
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = len(test_loader.dataset)

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    return epoch_loss, epoch_acc, running_corrects, total_samples


def main(args):
    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Prepare the data loaders
    train_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size)

    # Build and send the model to the device
    model = build_model('resnet50')
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = FSGDM(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, c_scaling=args.c_scaling, v_coefficient=args.v_coefficient, n_stages=args.n_stages, sigma=args.sigma)

    # Use CosineAnnealingLR so that the learning rate decreases following a cosine pattern over all epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        # Training phase
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluation phase on test set
        test_loss, test_acc, correct, total = evaluate(model, test_loader, criterion, device)

        # Print test results in the required format:
        print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {int(correct)}/{total} ({test_acc.item()*100:.2f}%)")

        # Save the best model based on test accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.save_path)

        # Step the cosine annealing scheduler (updates the learning rate)
        scheduler.step()

    print(f"Training complete. Best test accuracy: {best_acc.item()*100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ResNet-50 on CIFAR-100")
    parser.add_argument('--data_dir', default='./data', type=str,
                        help='Directory for CIFAR-100 dataset (default: ./data)')
    parser.add_argument('--epochs', default=300, type=int,
                        help='Number of training epochs (default: 300)')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Mini-batch size (default: 128)')
    parser.add_argument('--c_scaling', default=0.033, type=int,
                        help='Scaling factor for FSGDM (default: 0.033)')
    parser.add_argument('--v_coefficient', default=1.0, type=int,
                        help='Momentum coefficient for FSGDM (default: 1.0)')
    parser.add_argument('--n_stages', default=300, type=int,
                        help='Number of stages for FSGDM (default: 300)')
    parser.add_argument('--sigma', default=117000, type=int,
                        help='Number of gradient update steps (default: 117000)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate (default: 0.1)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum for SGD (default: 0.9)')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD (default: 5e-4)')
    parser.add_argument('--pretrained', action='store_true',
                        help='If set, use pretrained ResNet-50 weights (default: False)')
    parser.add_argument('--save_path', default='best_model.pth', type=str,
                        help='Path to save the best model (default: best_model.pth)')
    args = parser.parse_args()

    main(args)

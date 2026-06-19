"""
Train LeNet-5 on MNIST to >99% accuracy.
Saves the trained model for quantization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys

from lenet5_model import LeNet5

# Configuration
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
TARGET_ACCURACY = 0.99
MODEL_SAVE_PATH = 'float32_lenet5.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_mnist_loaders(batch_size=64):
    """Get MNIST train and test data loaders.
    MNIST images are 28x28, padded to 32x32 for LeNet-5 input.
    """
    transform = transforms.Compose([
        transforms.Pad(2),              # 28x28 -> 32x32 (pad 2 on each side)
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch {batch_idx+1}/{len(loader)}: '
                  f'Loss: {running_loss/(batch_idx+1):.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')

    return running_loss / len(loader), 100. * correct / total


def test(model, loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return test_loss / len(loader), 100. * correct / total


def main():
    print(f"Training LeNet-5 on MNIST")
    print(f"Device: {DEVICE}")
    print(f"Target accuracy: {TARGET_ACCURACY*100:.1f}%")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()

    # Data
    train_loader, test_loader = get_mnist_loaders(BATCH_SIZE)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print()

    # Model
    model = LeNet5().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        print("-" * 50)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        test_loss, test_acc = test(
            model, test_loader, criterion, DEVICE
        )

        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
        print()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_acc,
            }, MODEL_SAVE_PATH)
            print(f"  -> Best model saved (acc: {best_acc:.2f}%)")

        if test_acc >= TARGET_ACCURACY * 100:
            print(f"\n  Target accuracy {TARGET_ACCURACY*100:.1f}% reached at epoch {epoch}!")
            break

    print(f"\nTraining complete. Best accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {os.path.abspath(MODEL_SAVE_PATH)}")

    if best_acc < TARGET_ACCURACY * 100:
        print(f"\nWARNING: Did not reach target accuracy {TARGET_ACCURACY*100:.1f}%")
        print("Consider training for more epochs or tuning hyperparameters.")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())

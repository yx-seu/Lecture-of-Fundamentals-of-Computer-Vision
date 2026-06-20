"""
LeNet-5 Model Definition for MNIST
Standard LeNet-5 with 32x32 input (MNIST padded from 28x28)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """LeNet-5 for MNIST handwritten digit recognition.

    Architecture:
        Input: 1x32x32 (grayscale, padded from 28x28)
        C1: Conv2d(1, 6, 5) + ReLU  -> 6x28x28
        S2: MaxPool2d(2,2)           -> 6x14x14
        C3: Conv2d(6, 16, 5) + ReLU -> 16x10x10
        S4: MaxPool2d(2,2)           -> 16x5x5
        C5: Conv2d(16, 120, 5) + ReLU -> 120x1x1
        F6: Linear(120, 84) + ReLU   -> 84
        OUT: Linear(84, 10)          -> 10 (logits, no softmax)
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

        # Store layer output shapes for hardware mapping
        self.layer_shapes = {
            'input':  (1, 32, 32),
            'C1':     (6, 28, 28),
            'S2':     (6, 14, 14),
            'C3':     (16, 10, 10),
            'S4':     (16, 5, 5),
            'C5':     (120, 1, 1),
            'F6':     (84,),
            'OUT':    (10,),
        }

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier/Glorot initialization for weights, zero for biases."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # C1: Conv + ReLU
        x = F.relu(self.conv1(x))
        # S2: MaxPool
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # C3: Conv + ReLU
        x = F.relu(self.conv2(x))
        # S4: MaxPool
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # C5: Conv + ReLU (this conv produces 1x1 spatial)
        x = F.relu(self.conv3(x))
        # Flatten
        x = x.view(-1, 120)
        # F6: FC + ReLU
        x = F.relu(self.fc1(x))
        # OUT: FC (raw logits, no softmax for cross-entropy)
        x = self.fc2(x)
        return x

    def forward_with_layer_outputs(self, x):
        """Forward pass that also returns intermediate layer outputs.
        Useful for hardware verification.
        """
        outputs = {}

        x_in = x.clone()
        outputs['input'] = x_in

        x = F.relu(self.conv1(x))
        outputs['C1'] = x.clone()

        x = F.max_pool2d(x, kernel_size=2, stride=2)
        outputs['S2'] = x.clone()

        x = F.relu(self.conv2(x))
        outputs['C3'] = x.clone()

        x = F.max_pool2d(x, kernel_size=2, stride=2)
        outputs['S4'] = x.clone()

        x = F.relu(self.conv3(x))
        outputs['C5'] = x.clone()

        x_flat = x.view(-1, 120)
        x = F.relu(self.fc1(x_flat))
        outputs['F6'] = x.clone()

        x = self.fc2(x)
        outputs['OUT'] = x.clone()

        return outputs


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_layer_params(model):
    """Count parameters per layer (useful for BRAM allocation)."""
    counts = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            counts[name] = param.numel()
    return counts


if __name__ == '__main__':
    model = LeNet5()
    total, trainable = count_parameters(model)
    print(f"LeNet-5 Model Summary:")
    print(f"  Total parameters:    {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print()
    print("Layer parameter counts:")
    for name, count in count_layer_params(model).items():
        print(f"  {name:25s}: {count:>8,d}")
    print()
    print("Layer output shapes:")
    for name, shape in model.layer_shapes.items():
        if isinstance(shape, tuple):
            dim_str = "x".join(str(d) for d in shape)
            elements = 1
            for d in shape:
                elements *= d
            print(f"  {name:6s}: {dim_str:20s} = {elements:>6,d} elements")

    # Test forward pass
    dummy = torch.randn(1, 1, 32, 32)
    out = model(dummy)
    print(f"\n  Forward pass OK: input (1,1,32,32) -> output shape {out.shape}")

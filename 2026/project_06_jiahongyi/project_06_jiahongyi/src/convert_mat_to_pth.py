"""
Convert MatConvNet .mat DnCNN model to PyTorch .pth format.

The sigma=25.mat model has BN already merged into conv weights.
Architecture: Conv+ReLU → 15×(Conv+ReLU) → Conv  (17 conv layers total, 16 ReLUs)
"""
import h5py
import numpy as np
import torch
import torch.nn as nn
import sys
import os


class DnCNN(nn.Module):
    """DnCNN for grayscale image denoising.

    Architecture (17-layer, BN-merged):
        Conv(in=channels, out=64, k=3, bias=True) + ReLU
        15 × (Conv(in=64, out=64, k=3, bias=True) + ReLU)
        Conv(in=64, out=channels, k=3, bias=True)

    Residual learning: output = input - predicted_noise
    """

    def __init__(self, channels=1):
        super(DnCNN, self).__init__()
        self.channels = channels
        depth = 17
        n_channels = 64

        layers = []
        # First layer: Conv + ReLU (with bias, no BN — BN already merged)
        layers.append(nn.Conv2d(channels, n_channels, 3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        # Middle layers: 15 × (Conv + ReLU)
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, 3, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))
        # Last layer: Conv only, no activation
        layers.append(nn.Conv2d(n_channels, channels, 3, padding=1, bias=True))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise


def convert_mat_to_pth(mat_path, pth_path, verbose=True):
    """Convert a MatConvNet .mat model to PyTorch .pth checkpoint.

    Args:
        mat_path: Path to the .mat model file (e.g. sigma=25.mat)
        pth_path: Output path for the .pth checkpoint
        verbose: Print conversion details
    """
    f = h5py.File(mat_path, 'r')
    layers = f['net/layers']

    # Create model
    model = DnCNN(channels=1)
    state_dict = model.state_dict()

    # Map .mat layer weights to PyTorch state_dict keys
    # MatConvNet conv layers are at indices: 0, 2, 4, ..., 32 (every other index)
    pytorch_layer_names = []
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            pytorch_layer_names.append(name)

    if verbose:
        print(f"Model has {len(pytorch_layer_names)} parameter tensors")
        print(f"Mat file has {layers.shape[0]} layers")

    for i, pytorch_name in enumerate(pytorch_layer_names):
        # Each MatConvNet conv layer has 2 weight entries: [kernel, bias]
        mat_layer_idx = i // 2 * 2  # 0, 0, 2, 2, 4, 4, ...
        weight_entry_idx = i % 2    # 0 (kernel), 1 (bias)

        ref = layers[mat_layer_idx, 0]
        obj = f[ref]
        weights_refs = obj['weights']
        w_ref = weights_refs[weight_entry_idx, 0]
        w_data = f[w_ref][()]

        # Convert to torch tensor
        w_tensor = torch.from_numpy(np.array(w_data))

        # Handle shape conversion
        if weight_entry_idx == 0:  # kernel weights
            # MatConvNet format in this file: [Cout, Cin, H, W] — already PyTorch format
            # But singleton dimensions may be squeezed by h5py
            if w_tensor.ndim == 3:
                # Last layer: (64, 3, 3) → (1, 64, 3, 3) for Cout=1
                # Actually it could be (1, 64, 3, 3) with Cout=1 squeezed
                # Or it could be (64, 1, 3, 3) with Cin=1 squeezed
                # For the last layer: Cin=64, Cout=1 → we need (1, 64, 3, 3)
                # For the first layer: Cin=1, Cout=64 → we'd need (64, 1, 3, 3)
                if pytorch_name.endswith('dncnn.0.weight'):
                    # First conv: Cin=1, Cout=64 → (64, 1, 3, 3)
                    w_tensor = w_tensor.view(64, 1, 3, 3)
                else:
                    # Last conv: Cin=64, Cout=1 → (1, 64, 3, 3)
                    w_tensor = w_tensor.view(1, 64, 3, 3)
        else:  # bias
            # Bias stored as (1, Cout) → flatten to (Cout,)
            w_tensor = w_tensor.view(-1)

        # Verify shape matches
        expected_shape = state_dict[pytorch_name].shape
        if w_tensor.shape != expected_shape:
            # Try alternative reshaping
            if verbose:
                print(f"  Shape mismatch for {pytorch_name}: got {w_tensor.shape}, expected {expected_shape}")
            # Try to reshape
            w_tensor = w_tensor.reshape(expected_shape)

        state_dict[pytorch_name] = w_tensor

        if verbose:
            name_ds = obj['name']
            name_arr = name_ds[()]
            mat_name = ''.join(chr(x[0]) for x in name_arr) if name_arr.ndim > 1 else ''.join(chr(x) for x in name_arr)
            print(f"  {pytorch_name:40s} ← {mat_name.strip()}[{weight_entry_idx}] shape={w_tensor.shape}")

    # Load into model and save
    model.load_state_dict(state_dict)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'channels': 1,
        'depth': 17,
        'n_channels': 64,
    }
    torch.save(checkpoint, pth_path)
    print(f"\nSaved PyTorch checkpoint to: {pth_path}")

    f.close()

    # Quick sanity check
    model.eval()
    test_input = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        test_output = model(test_input)
    print(f"Sanity check: input shape={test_input.shape}, output shape={test_output.shape}")
    print("Conversion successful!")

    return model


if __name__ == '__main__':
    mat_path = sys.argv[1] if len(sys.argv) > 1 else '../../DnCNN-master/model/specifics/sigma=25.mat'
    pth_path = sys.argv[2] if len(sys.argv) > 2 else 'dncnn_pretrained.pth'
    convert_mat_to_pth(mat_path, pth_path)

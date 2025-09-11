import os
import numpy as np
import torch
from torch.utils.data import Dataset

class Extract(Dataset):
    def __init__(self, data_dir, label_dir, subpatch_size=(16, 64, 64), stride_ratio=(0.5, 0.5, 0.5)):
        self.data_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')])
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.npy')])
        self.subpatch_size = subpatch_size
        self.stride = tuple(int(sz * sr) for sz, sr in zip(subpatch_size, stride_ratio))
        self.data, self.labels = self._load_and_preprocess_data()
        self.subpatch_indices = self._prepare_subpatch_indices()

    def _load_and_preprocess_data(self):
        """Load and preprocess the data and labels (normalize + binarize)."""
        all_data, all_labels = [], []
        for data_path, label_path in zip(self.data_paths, self.label_paths):
            data_patch = np.load(data_path)
            label_patch = np.load(label_path)

            # Apply Z-score normalization
            mean, std = np.mean(data_patch), np.std(data_patch) + 1e-8
            data_patch = (data_patch - mean) / std

            # Binarize the label
            label_patch = (label_patch > 0).astype(np.float32)

            all_data.append(data_patch)
            all_labels.append(label_patch)

        return all_data, all_labels

    def _prepare_subpatch_indices(self):
        """Generate sub-patch indices from all loaded patches."""
        subpatch_indices = []
        for idx, data_patch in enumerate(self.data):
            positions = self._get_subpatch_positions(data_patch.shape)
            subpatch_indices.extend([(idx, pos) for pos in positions])
        return subpatch_indices

    def _get_subpatch_positions(self, patch_shape):
        """Calculate all possible sub-patch positions."""
        z_size, y_size, x_size = self.subpatch_size
        z_stride, y_stride, x_stride = self.stride
        positions = [
            (z, y, x)
            for z in range(0, patch_shape[0] - z_size + 1, z_stride)
            for y in range(0, patch_shape[1] - y_size + 1, y_stride)
            for x in range(0, patch_shape[2] - x_size + 1, x_stride)
        ]
        return positions

    def __len__(self):
        return len(self.subpatch_indices)

    def __getitem__(self, index):
        """Retrieve a specific sub-patch."""
        patch_idx, (z, y, x) = self.subpatch_indices[index]

        data_patch = self.data[patch_idx]
        label_patch = self.labels[patch_idx]

        data_subpatch = data_patch[z:z + self.subpatch_size[0], 
                                   y:y + self.subpatch_size[1], 
                                   x:x + self.subpatch_size[2]]

        label_subpatch = label_patch[z:z + self.subpatch_size[0], 
                                     y:y + self.subpatch_size[1], 
                                     x:x + self.subpatch_size[2]]

        # Add channel dimension (unsqueeze)
        data_subpatch = np.expand_dims(data_subpatch, axis=0)  # (1, Z, Y, X)
        label_subpatch = np.expand_dims(label_subpatch, axis=0)  # (1, Z, Y, X)

        # Convert to PyTorch tensors
        data_subpatch = torch.tensor(data_subpatch, dtype=torch.float32)
        label_subpatch = torch.tensor(label_subpatch, dtype=torch.float32)

        return data_subpatch, label_subpatch

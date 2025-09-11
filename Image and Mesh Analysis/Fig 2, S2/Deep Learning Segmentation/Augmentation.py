import torch
import numpy as np
import random
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

class AugmentData:
    def __init__(self, em_image, label, mask, prob_flip_rotation=1.0, prob_noise=0.5, 
                 prob_blur=0.5, prob_missing=0.5, blur_sigma=1.0):
        """
        Initialize the AugmentData class with the given EM image, label, mask, and probabilities.
        :param em_image: Input 3D EM image array (e.g., (1, Z, Y, X))
        :param label: Corresponding 3D label array (same shape as em_image)
        :param mask: Corresponding mask indicating valid regions
        :param prob_flip_rotation: Probability of applying flip/rotation (default 1.0)
        :param prob_noise: Probability of adding Gaussian noise (default 0.5)
        :param prob_blur: Probability of applying Gaussian blur (default 0.5)
        :param prob_missing: Probability of introducing a missing section (default 0.5)
        """
        self.em_image = em_image
        self.label = label
        self.mask = mask
        self.prob_flip_rotation = prob_flip_rotation
        self.prob_noise = prob_noise
        self.prob_blur = prob_blur
        self.prob_missing = prob_missing
        self.blur_sigma = blur_sigma

    def apply_flip_and_rotation(self, data, label, mask):
        """Apply random flip to EM image, label, and mask."""
        flip_axes = [axis for axis in range(1, 4) if random.choice([True, False])]
        for axis in flip_axes:
            data = np.flip(data, axis=axis).copy()  # Copy to avoid negative strides
            label = np.flip(label, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()

        return data, label, mask

    def add_gaussian_noise(self, data):
        """Add Gaussian noise to the EM image."""
        if random.random() < self.prob_noise:
            noise = np.random.normal(0, 0.1, data.shape)
            data += noise
        return data

    def add_gaussian_blur(self, data):
        """Apply Gaussian blur across the entire 3D stack."""
        if random.random() < self.prob_blur:
            data = gaussian_filter(data, sigma=self.blur_sigma)
        return data

    def add_missing_slices(self, data, label, mask):
        """Add 1 to 3 consecutive missing slices, filled with the mean of surrounding slices."""
        if random.random() < self.prob_missing:
            num_missing_slices = random.randint(1, 3)
            z_idx = random.randint(1, data.shape[1] - num_missing_slices - 1)
            mean_slice = (data[:, z_idx - 1, :, :] + data[:, z_idx + num_missing_slices, :, :]) / 2

            for i in range(num_missing_slices):
                data[:, z_idx + i, :, :] = mean_slice
                label[:, z_idx + i, :, :] = (label[:, z_idx - 1, :, :] + label[:, z_idx + num_missing_slices, :, :]) / 2
                mask[:, z_idx + i, :, :] = (mask[:, z_idx - 1, :, :] + mask[:, z_idx + num_missing_slices, :, :]) / 2

        return data, label, mask

    def apply_augmentations(self):
        """Apply all augmentations to the EM image, label, and mask."""
        augmented_image, augmented_label, augmented_mask = self.apply_flip_and_rotation(
            self.em_image.copy(), self.label.copy(), self.mask.copy()
        )
        augmented_image = self.add_gaussian_noise(augmented_image)
        augmented_image = self.add_gaussian_blur(augmented_image)
        augmented_image, augmented_label, augmented_mask = self.add_missing_slices(augmented_image, augmented_label, augmented_mask)
        
        return augmented_image, augmented_label, augmented_mask


class AugmentedDataset(Dataset):
    def __init__(self, dataset, augment=True, blur_sigma=1.0):
        """
        Wrapper to apply augmentations to data, label, and mask.
        :param dataset: Original dataset (e.g., PaddedExtract)
        :param augment: Whether to apply augmentations (for training only)
        """
        self.dataset = dataset
        self.augment = augment
        self.blur_sigma = blur_sigma

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the original EM image, label, and mask
        em_image, label, mask = self.dataset[idx]

        if self.augment:
            augmenter = AugmentData(
                em_image.numpy(), label.numpy(), mask.numpy(),
                prob_flip_rotation=1.0, prob_noise=0.5, prob_blur=0.5, prob_missing=0.25,
                blur_sigma=self.blur_sigma
            )
            em_image, label, mask = augmenter.apply_augmentations()

            # Convert back to tensors
            em_image = torch.tensor(em_image, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)

        return em_image, label, mask  # Return all three elements

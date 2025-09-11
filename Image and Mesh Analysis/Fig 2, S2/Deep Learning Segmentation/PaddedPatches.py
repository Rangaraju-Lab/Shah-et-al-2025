import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class PaddedExtract:
    """
    Wrapper class to apply swap, pad, and mask operations on Extract patches.
    """

    def __init__(self, extractor, target_shape=(16, 64, 64), pad_mode='reflect'):
        """
        Initialize with an extractor object.
        :param extractor: Extract object containing original data and labels.
        :param target_shape: Desired shape for the output patches (Z, Y, X).
        :param pad_mode: Padding mode (default is 'reflect').
        """
        self.extractor = extractor
        self.target_shape = target_shape
        self.pad_mode = pad_mode

    def swap_and_pad(self, patch):
        """Perform swapping, padding, and mask generation for a given patch."""
        patch = patch.squeeze()  # Remove batch and channel dimensions
        original_shape = patch.shape

        # Determine the permutation order to make shape ascending (e.g., (32, 16, 64) -> (16, 32, 64))
        permute_order = sorted(range(len(original_shape)), key=lambda i: original_shape[i])
        patch = patch.permute(*permute_order)

        # Calculate padding along the Y-axis (middle dimension)
        y_padding = (self.target_shape[1] - patch.shape[1]) // 2

        # Create mask (1s for valid regions, 0s for padded regions)
        mask = torch.ones_like(patch, dtype=torch.float32)

        if y_padding > 0:
            # Apply padding to both patch and mask
            patch = F.pad(patch, (0, 0, y_padding, y_padding), mode=self.pad_mode)
            mask = F.pad(mask, (0, 0, y_padding, y_padding), mode='constant', value=0)

        # Ensure the final shape matches the target shape
        assert patch.shape == self.target_shape, f"Expected shape {self.target_shape}, got {patch.shape}"

        return patch.unsqueeze(0), mask.unsqueeze(0)  # Add batch dimension back

    def __len__(self):
        """Return the number of patches in the extractor."""
        return len(self.extractor)

    def __getitem__(self, idx):
        """Retrieve a patch with the corresponding padded label and mask."""
        data_patch, label_patch = self.extractor[idx]  # Original patch

        # Apply swap, pad, and mask generation to both data and label
        padded_data, data_mask = self.swap_and_pad(data_patch)
        padded_label, _ = self.swap_and_pad(label_patch)  # Mask not needed for label

        return padded_data, padded_label, data_mask

    def display_patch(self, idx):
        """
        Display the data, label, and mask for a given patch index.
        :param idx: Index of the patch to display.
        """
        # Retrieve the padded data, label, and mask
        data, label, mask = self[idx]

        # Extract the first slice along Z for visualization
        data_slice = data[0, 0, :, :].numpy()
        label_slice = label[0, 0, :, :].numpy()
        mask_slice = mask[0, 0, :, :].numpy()

        # Plot the slices side by side
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['Data', 'Label', 'Mask']

        # Define custom scaling per image type
        for ax, img, title in zip(axes, [data_slice, label_slice, mask_slice], titles):
            if title == 'Data':
                ax.imshow(img, cmap='gray')  # Auto-scale for data
            else:
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)  # Fixed scaling for mask/label

            ax.set_title(f"{title} (Patch {idx})")
            ax.axis('off')

        plt.show()

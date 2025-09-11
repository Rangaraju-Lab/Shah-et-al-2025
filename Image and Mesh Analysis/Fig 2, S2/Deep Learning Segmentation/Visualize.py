import os 
import torch 
import numpy as np 
import matplotlib.pyplot as plt

def save_slices(em_image, prediction, epoch, train_loss, val_loss, output_dir, prefix="inference"):
    """Save the first, middle, and last slices of EM data and prediction with losses in the subtitle."""

    # Ensure em_image and prediction are NumPy arrays
    if isinstance(em_image, torch.Tensor):
        em_image = em_image.squeeze().cpu().numpy()  # Convert to NumPy and remove batch/channel dim
    else:
        em_image = em_image.squeeze()  # Remove batch/channel dimension if it's a NumPy array

    if isinstance(prediction, torch.Tensor):
        prediction = prediction.squeeze().cpu().numpy()  # Convert to NumPy and remove batch/channel dim
    else:
        prediction = prediction.squeeze()  # Remove batch/channel dimension if it's a NumPy array

    # Extract valid 2D slices along the Z-axis
    first_slice, mid_slice, last_slice = 0, em_image.shape[0] // 2, em_image.shape[0] - 1
    titles = ["First Slice", "Middle Slice", "Last Slice"]

    # Create figure and axes (2 rows: EM Image + Prediction)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns

    # Plot EM Image slices
    for i, slice_idx in enumerate([first_slice, mid_slice, last_slice]):
        axes[0, i].imshow(em_image[slice_idx], cmap='gray')  # Select 2D slice
        axes[0, i].set_title(f"EM Image - {titles[i]}", fontsize=12)
        axes[0, i].axis('off')

    # Plot Prediction slices
    for i, slice_idx in enumerate([first_slice, mid_slice, last_slice]):
        axes[1, i].imshow(prediction[slice_idx], cmap='gray')  # Select 2D slice
        axes[1, i].set_title(f"Prediction - {titles[i]}", fontsize=12)
        axes[1, i].axis('off')

    # Add subtitle with average losses
    fig.suptitle(
        f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}",
        fontsize=16, y=0.95
    )

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    save_path = os.path.join(output_dir, f"{prefix}_epoch_{epoch}.tif")
    plt.savefig(save_path, format='tiff', dpi=300)
    plt.close()

    print(f"Saved inference results to {save_path}")
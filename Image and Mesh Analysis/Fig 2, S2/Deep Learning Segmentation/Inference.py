import numpy as np
from tifffile import imread, imwrite
import torch
from itertools import product
from scipy.ndimage import rotate
from tqdm import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time
from ResidualUNet import ResidualUNet
from tqdm import tqdm
import os

class Preprocess:
    def __init__(self, patch_size=(32, 128, 128), stride=(16, 64, 64), device='cuda'):
        self.patch_size = patch_size
        self.stride = stride
        self.device = torch.device(device)

    def load_and_swap_axes(self, filepath, axis_order=(0, 1, 2)):
        """Load and z-score normalize a .tif image after swapping axes."""
        image = imread(filepath).astype(np.float32)
        swapped_image = np.transpose(image, axes=axis_order)
        mean, std = swapped_image.mean(), swapped_image.std() + 1e-8
        print(swapped_image.shape)
        return (swapped_image - mean) / std

    def center_pad_image(self, image):
        """Center-pad an image to make dimensions a multiple of patch_size."""
        original_shape = image.shape
        pad_width = [(p // 2, p - p // 2) for p in [
            (self.patch_size[i] - (image.shape[i] % self.patch_size[i])) % self.patch_size[i] for i in range(3)
        ]]
        padded_image = np.pad(image, pad_width=pad_width, mode='reflect')
        return padded_image, original_shape, pad_width

    def extract_patches(self, image, stride=None):
        """Extract overlapping patches from an image with specified stride on GPU."""
        
        # Use the stride argument if provided; otherwise, use the instance's default stride
        if stride is None:
            stride = self.stride

        # Move the padded image to GPU
        image_gpu = torch.tensor(image, device=self.device)

        z_size, y_size, x_size = self.patch_size
        z_stride, y_stride, x_stride = stride
        z_range = range(0, image_gpu.shape[0] - z_size + 1, z_stride)
        y_range = range(0, image_gpu.shape[1] - y_size + 1, y_stride)
        x_range = range(0, image_gpu.shape[2] - x_size + 1, x_stride)

        patches = []
        positions = []

        # Extract patches directly on GPU
        for z in z_range:
            for y in y_range:
                for x in x_range:
                    patch = image_gpu[z:z + z_size, y:y + y_size, x:x + x_size]
                    patches.append(patch)
                    positions.append((z, y, x))

        # Stack patches into a single tensor on GPU
        patches_tensor = torch.stack(patches)

        return patches_tensor, positions









class Inference:
    def __init__(self, model_path, device='cuda', batch_size=12, patch_size=(32, 128, 128)):
        self.device = torch.device(device)
        self.model = self.load_model(model_path, self.device)
        self.batch_size = batch_size
        self.patch_size = patch_size

    def load_model(self, model_path, device):
        model = ResidualUNet().to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        model.eval()
        return model

    def prepare_batches(self, patches):
        """Prepare patches by adding a channel dimension and sending them to the GPU."""
        if patches.dim() == 4:  # Ensure patches have dimensions (batch, D, H, W)
            patches = patches.unsqueeze(1)  # Add channel dimension to make (batch, C, D, H, W)
        return patches.float()

    def generate_transformations(self):
        flip_combinations = list(product([0, 1], repeat=3))
        rotation_angles = [0, 90]
        return [(flip_axes, angle) for flip_axes in flip_combinations for angle in rotation_angles]

    def apply_augmentation(self, batch, flip_axes, rotation_angle):
        for axis, do_flip in enumerate(flip_axes, start=2):
            if do_flip:
                batch = torch.flip(batch, dims=(axis,))
        if rotation_angle != 0:
            batch = torch.rot90(batch, k=rotation_angle // 90, dims=(3, 4))
        return batch

    def reverse_augmentation(self, batch, flip_axes, rotation_angle):
        if rotation_angle != 0:
            batch = torch.rot90(batch, k=-(rotation_angle // 90), dims=(3, 4))
        for axis, do_flip in enumerate(flip_axes, start=2):
            if do_flip:
                batch = torch.flip(batch, dims=(axis,))
        return batch

    def batch_infer(self, batch):
        with torch.no_grad():
            return self.model(batch)

    def infer_with_augmentations(self, patches, transfer_interval=1):
        transformations = self.generate_transformations()
        total_batches = len(patches) // self.batch_size
        gpu_aggregated_predictions = []
        final_predictions = []

        with tqdm(total=total_batches+1, desc="Overall Progress", dynamic_ncols=False, ascii=True) as pbar:
            for i in range(0, len(patches), self.batch_size):
                batch = self.prepare_batches(patches[i:i + self.batch_size])
                batch_predictions = []

                for aug_idx, (flip_axes, rotation_angle) in enumerate(transformations):
                    aug_batch = self.apply_augmentation(batch, flip_axes, rotation_angle)
                    pred_batch = self.batch_infer(aug_batch)  # Inference on GPU
                    reversed_batch = self.reverse_augmentation(pred_batch, flip_axes, rotation_angle)
                    batch_predictions.append(reversed_batch)

                gpu_aggregated_predictions.append(torch.median(torch.stack(batch_predictions, dim=0), dim=0).values)
                pbar.update(1)

                if len(gpu_aggregated_predictions) >= transfer_interval:
                    final_predictions.append(torch.cat(gpu_aggregated_predictions, dim=0).cpu().numpy())
                    gpu_aggregated_predictions = []

            if gpu_aggregated_predictions:
                final_predictions.append(torch.cat(gpu_aggregated_predictions, dim=0).cpu().numpy())

        return np.concatenate(final_predictions, axis=0)

    def run_inference_on_patches(self, patches):
        # Run inference with augmentations; patches are expected to be on GPU with correct dimensions
        return self.infer_with_augmentations(patches)


class Postprocess:
    def __init__(self, patch_size=(32, 128, 128), t=3):
        self.patch_size = patch_size
        self.bump_mask = self.create_bump_function(t)

    def create_bump_function(self, t=3):
        coords = [np.linspace(-1, 1, size) for size in self.patch_size]
        grids = np.meshgrid(*coords, indexing="ij")
        bump = np.exp(-np.sum([(grid ** 2) * t for grid in grids], axis=0))
        return bump / np.max(bump)

    def reassemble_patches(self, patches, positions):
        max_z = max(pos[0] for pos in positions) + self.patch_size[0]
        max_y = max(pos[1] for pos in positions) + self.patch_size[1]
        max_x = max(pos[2] for pos in positions) + self.patch_size[2]
        output_image, weight_map = np.zeros((max_z, max_y, max_x), dtype=np.float32), np.zeros((max_z, max_y, max_x), dtype=np.float32)

        for patch, (z, y, x) in zip(patches, positions):
            weighted_patch = patch[0] * self.bump_mask
            output_image[z:z + self.patch_size[0], y:y + self.patch_size[1], x:x + self.patch_size[2]] += weighted_patch
            weight_map[z:z + self.patch_size[0], y:y + self.patch_size[1], x:x + self.patch_size[2]] += self.bump_mask

        # Option to normalize to average overlapping regions
        output_image = np.divide(output_image, weight_map, where=(weight_map != 0))
        return output_image

    def unpad_image(self, padded_image, original_shape, pad_width):
        """Remove padding to restore the image's original shape."""
        slices = tuple(slice(pad[0], padded_image.shape[i] - pad[1]) for i, pad in enumerate(pad_width))
        return padded_image[slices]

    def unswap_axes_and_save(self, image, axis_order=(0, 1, 2), output_filepath="./inference_output/output_image.tif"):
        inverse_order = np.argsort(axis_order)
        unswapped_image = np.transpose(image, axes=inverse_order)
        imwrite(output_filepath, unswapped_image.astype(np.float32))
        print(f"Image saved as {output_filepath}")
        return unswapped_image




import numpy as np
from tifffile import imread, imwrite

def merge_tifs_with_average(filepaths, output_filepath="merged_output.tif"):
    """
    Load three .tif files, compute the voxel-wise average, and save the result.
    
    Parameters:
    - filepaths: list of str, paths to the three .tif files to be merged
    - output_filepath: str, path to save the merged .tif file
    
    Returns:
    - averaged_image: np.array, the voxel-wise averaged image
    """
    # Check if exactly three files are provided
    if len(filepaths) != 3:
        raise ValueError("Exactly three .tif file paths are required.")
    
    # Load each .tif file as a NumPy array
    images = [imread(filepath).astype(np.float32) for filepath in filepaths]
    
    # Ensure all images have the same shape
    if not all(img.shape == images[0].shape for img in images):
        raise ValueError("All .tif files must have the same shape.")
    
    # Compute the voxel-wise average
    averaged_image = np.mean(images, axis=0)
    
    # Save the averaged image
    imwrite(output_filepath, averaged_image.astype(np.float32))
    print(f"Averaged image saved as {output_filepath}")
    
    return averaged_image


        
# Ensure the output directory exists
os.makedirs("./inference_output", exist_ok=True)


patch_size = (32, 128, 128)
stride=(16, 64, 64)
# Process XY plane
preprocessor = Preprocess(patch_size=patch_size, stride=stride)
swapped_image = preprocessor.load_and_swap_axes("Control4.tif", axis_order=(0, 1, 2))
padded_image, original_shape, pad_width = preprocessor.center_pad_image(swapped_image)
patches, positions = preprocessor.extract_patches(padded_image)

inference = Inference(model_path="./models/model295.pth", device='cuda')
predictions = inference.run_inference_on_patches(patches)

postprocessor = Postprocess(patch_size=patch_size)
reassembled_image = postprocessor.reassemble_patches(predictions, positions)
unpadded_image = postprocessor.unpad_image(reassembled_image, original_shape, pad_width)
unswapped_image = postprocessor.unswap_axes_and_save(unpadded_image, axis_order=(0, 1, 2), output_filepath="./inference_output/Control4XY.tif")



patch_size = (128, 32, 128)
stride=(64, 16, 64)
# Process XZ plane
preprocessor = Preprocess(patch_size=patch_size, stride=stride)
swapped_image = preprocessor.load_and_swap_axes("Control4.tif", axis_order=(1, 0, 2))
padded_image, original_shape, pad_width = preprocessor.center_pad_image(swapped_image)
patches, positions = preprocessor.extract_patches(padded_image)

inference = Inference(model_path="./models/model295.pth", device='cuda')
predictions = inference.run_inference_on_patches(patches)

postprocessor = Postprocess(patch_size=patch_size)
reassembled_image = postprocessor.reassemble_patches(predictions, positions)
unpadded_image = postprocessor.unpad_image(reassembled_image, original_shape, pad_width)
unswapped_image = postprocessor.unswap_axes_and_save(unpadded_image, axis_order=(1, 0, 2), output_filepath="./inference_output/Control4XZ.tif")

patch_size = (128, 128, 32)
stride=(64, 64, 16)
# Process YZ plane
preprocessor = Preprocess(patch_size=patch_size, stride=stride)
swapped_image = preprocessor.load_and_swap_axes("Control4.tif", axis_order=(2, 1, 0))
padded_image, original_shape, pad_width = preprocessor.center_pad_image(swapped_image)
patches, positions = preprocessor.extract_patches(padded_image)

inference = Inference(model_path="./models/model295.pth", device='cuda')
predictions = inference.run_inference_on_patches(patches)

postprocessor = Postprocess(patch_size=patch_size)
reassembled_image = postprocessor.reassemble_patches(predictions, positions)
unpadded_image = postprocessor.unpad_image(reassembled_image, original_shape, pad_width)
unswapped_image = postprocessor.unswap_axes_and_save(unpadded_image, axis_order=(2, 1, 0), output_filepath="./inference_output/Control4YZ.tif")

# Merge all predictions
filepaths = ["./inference_output/Control4XY.tif", "./inference_output/Control4XZ.tif", "./inference_output/Control4YZ.tif"]
merged_image = merge_tifs_with_average(filepaths, output_filepath="./inference_output/Prediction_Control4.tif")
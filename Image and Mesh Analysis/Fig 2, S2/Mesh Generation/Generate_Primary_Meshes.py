import os
import numpy as np
import tifffile as tiff
from scipy.ndimage import label
from skimage.morphology import remove_small_objects

def process_tiff_component(file_path, min_voxel_size, extrude=True, num_extrude=10):
    """
    Processes a TIFF file by:
    1. Extruding the TIFF stack (if enabled).
    2. Extracting connected components.
    3. Filtering out components below `min_voxel_size`.
    4. Merging remaining components into a single binary volume.
    5. Extruding the borders by copying edge pixels.

    Parameters:
        file_path (str): Path to the TIFF file.
        min_voxel_size (int): Minimum voxel count for a component to be included.
        extrude (bool): If True, extrudes the TIFF stack before processing.
        num_extrude (int): Number of slices to duplicate at the top, bottom, and borders.

    Returns:
        np.ndarray: A single 3D NumPy array representing the merged component.
    """
    # Load the TIFF stack
    image_stack = tiff.imread(file_path)
    binary_stack = image_stack > 0  # Convert to binary mask

    # Step 1: Remove small objects before labeling
    filtered_stack = remove_small_objects(binary_stack.astype(bool), min_size=min_voxel_size)
    filtered_stack = filtered_stack.astype(np.uint8)  # Convert back to binary

    # Step 2: Extract connected components and merge them into a single volume
    labeled_stack, num_features = label(filtered_stack)
    combined_stack = np.zeros_like(filtered_stack, dtype=np.uint8)

    for i in range(1, num_features + 1):
        component = (labeled_stack == i).astype(np.uint8)
        if np.sum(component) >= min_voxel_size:  # Keep only large components
            combined_stack |= component  # Merge components into one

    # Step 3: Extrude in the Z direction (top & bottom)
    if extrude:
        top_slice = combined_stack[0:1, :, :]
        bottom_slice = combined_stack[-1:, :, :]
        combined_stack = np.vstack([
            np.repeat(top_slice, num_extrude, axis=0),
            combined_stack,
            np.repeat(bottom_slice, num_extrude, axis=0)
        ])

    # Step 4: Extrude borders correctly (after height extrusion)
    def extrude_borders(stack, num_extrude):
        # Extract updated borders AFTER height extrusion
        left_border = stack[:, :, 0:1]
        right_border = stack[:, :, -1:]
        front_border = stack[:, 0:1, :]
        back_border = stack[:, -1:, :]

        # Apply border extrusion correctly
        stack = np.concatenate([
            np.repeat(front_border, num_extrude, axis=1),  # Front
            stack,
            np.repeat(back_border, num_extrude, axis=1)   # Back
        ], axis=1)

        # Extract left/right borders again after height extrusion
        left_border = stack[:, :, 0:1]
        right_border = stack[:, :, -1:]

        stack = np.concatenate([
            np.repeat(left_border, num_extrude, axis=2),  # Left
            stack,
            np.repeat(right_border, num_extrude, axis=2)  # Right
        ], axis=2)

        return stack

    combined_stack = extrude_borders(combined_stack, num_extrude)

    return combined_stack





import numpy as np
import tifffile as tiff



def bounding_box(file_path, pad_size=10):
    """
    Converts a TIFF stack to a binary mask (all ones), then pads it with zeros 
    on the top, bottom, and borders (X, Y, and Z) for a chosen number of pixels.

    Parameters:
        file_path (str): Path to the TIFF file.
        pad_size (int): Number of pixels to pad on each side.

    Returns:
        np.ndarray: A single 3D NumPy array representing the padded component.
    """
    # Load TIFF stack
    image_stack = tiff.imread(file_path)

    # Convert to binary mask (all ones)
    binary_stack = np.ones_like(image_stack, dtype=np.uint8)

    # Apply zero-padding to all sides (Z, Y, X)
    padded_stack = np.pad(binary_stack, pad_width=pad_size, mode='constant', constant_values=0)

    return padded_stack  # Return single padded component






from skimage.measure import marching_cubes
import trimesh
import numpy as np



import numpy as np
import trimesh
from skimage.measure import marching_cubes

def component_to_mesh(combined_stack, add_empty_slices=True, add_empty_border=True):
    """
    Converts a single 3D binary component into a 3D mesh using the 
    Marching Cubes algorithm. Optionally adds an empty slice before 
    and after, and a one-pixel empty border around all sides.

    Parameters:
        combined_stack (np.ndarray): 3D NumPy array representing the component.
        add_empty_slices (bool): If True, adds an empty slice before and after.
        add_empty_border (bool): If True, adds a one-pixel empty border.

    Returns:
        trimesh.Trimesh: The reconstructed 3D mesh.
    """
    # Ensure float values for Marching Cubes
    combined_stack = combined_stack.astype(np.float32)

    # Add empty slices if enabled
    if add_empty_slices:
        empty_slice = np.zeros((1, combined_stack.shape[1], combined_stack.shape[2]), dtype=np.float32)
        combined_stack = np.vstack([empty_slice, combined_stack, empty_slice])

    # Add a one-pixel empty border if enabled
    if add_empty_border:
        combined_stack = np.pad(combined_stack, pad_width=1, mode='constant', constant_values=0)

    # Determine a valid threshold for Marching Cubes
    level = 0.5 

    # Generate the mesh
    verts, faces, normals, _ = marching_cubes(combined_stack, level=level)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    return mesh


import trimesh


base_dir = r"./Demo Images/control 4"


filename = "OMM.tiff"
file_path = os.path.join(base_dir, filename)
min_voxel_size = 50  # Minimum voxel threshold
component = process_tiff_component(file_path, min_voxel_size, extrude=True)
component = component_to_mesh(component)
#component_meshes = smooth_meshes(component_meshes, iterations=10, alpha=0.2, beta=0.3)
micron_per_pixel = 0.002 
component.apply_scale(micron_per_pixel)
output = "OMM.stl"
output_path = os.path.join(base_dir, output)
component.export(output_path)

filename = "IBM.tiff"
file_path = os.path.join(base_dir, filename)
min_voxel_size = 50  # Minimum voxel threshold
component = process_tiff_component(file_path, min_voxel_size, extrude=True, num_extrude=10)
component = component_to_mesh(component)
micron_per_pixel = 0.002 
component.apply_scale(micron_per_pixel)
output = "IBM.stl"
output_path = os.path.join(base_dir, output)
component.export(output_path)

filename = "IMS.tiff"
file_path = os.path.join(base_dir, filename)
min_voxel_size = 50  # Minimum voxel threshold
component = process_tiff_component(file_path, min_voxel_size, extrude=True, num_extrude=15)
component = component_to_mesh(component)
micron_per_pixel = 0.002 
component.apply_scale(micron_per_pixel)
output = "IMS.stl"
output_path = os.path.join(base_dir, output)
component.export(output_path)

filename = "ER.tiff"
file_path = os.path.join(base_dir, filename)
min_voxel_size = 50  # Minimum voxel threshold
component = process_tiff_component(file_path, min_voxel_size, extrude=True)
component = component_to_mesh(component)
micron_per_pixel = 0.002 
component.apply_scale(micron_per_pixel)
output = "ER.stl"
output_path = os.path.join(base_dir, output)
component.export(output_path)

filename = "Ribo.tiff"
file_path = os.path.join(base_dir, filename)
min_voxel_size = 50  # Minimum voxel threshold
component = process_tiff_component(file_path, min_voxel_size, extrude=True)
component = component_to_mesh(component)
micron_per_pixel = 0.002 
component.apply_scale(micron_per_pixel)
output = "Ribo.stl"
output_path = os.path.join(base_dir, output)
component.export(output_path)

component = bounding_box(file_path)
component = component_to_mesh(component)
micron_per_pixel = 0.002 
component.apply_scale(micron_per_pixel)
output = "BB.stl"
output_path = os.path.join(base_dir, output)
component.export(output_path)

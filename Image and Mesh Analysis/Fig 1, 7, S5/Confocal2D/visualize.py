import matplotlib.pyplot as plt
import numpy as np

def direction_map(
    med_image, dendrites, save_image=False, output_filename="Dendritic_Direction.tif", dpi=300, arrow_size=10, inset_fraction=0.05
):
    """
    Visualize the medial axis with polylines and direction arrows placed slightly inward, and optionally save as a TIFF.
    The plot dimensions match the image dimensions with no titles or boundaries.

    Parameters:
    - med_image: Background image (2D numpy array).
    - dendrites: Dictionary of dendrite polylines (Branch_ID -> list of (X, Y) coordinates).
    - save_image: Boolean flag to save the plot as a .tif file. Default is False.
    - output_filename: The filename for saving the image. Default is "Dendritic_Direction.tif".
    - dpi: Resolution in dots per inch for saving the image. Default is 300.
    - arrow_size: Size of the arrow heads for start and end points. Default is 10.
    - inset_fraction: Fraction of the polyline length to inset arrows. Default is 0.1.
    """
    if len(med_image.shape) > 2:
        raise ValueError("Input image must be a 2D array.")
    
    # Compute figure size to match image dimensions
    fig, ax = plt.subplots(figsize=(med_image.shape[1] / dpi, med_image.shape[0] / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
    fig.patch.set_facecolor('black')  # Set figure background color
    ax.set_facecolor('black')        # Set axes background color
    ax.imshow(med_image, cmap='gray', interpolation='none', origin='upper')  # Show the medial axis image
    ax.axis("off")  # Turn off axis lines and labels

    # Plot the polylines and arrows
    for branch_id, polyline in dendrites.items():
        polyline = np.array(polyline)  # Ensure polyline is a NumPy array
        num_points = len(polyline)

        # Plot the polyline in yellow
        ax.plot(polyline[:, 0], polyline[:, 1], color='yellow', linewidth=2, zorder=1, alpha=0.25)

        if num_points > 2:
            # Determine inset index for start and end
            inset_start_idx = max(1, int(inset_fraction * num_points))
            inset_end_idx = max(1, int((1 - inset_fraction) * num_points))

            # Start arrow
            start_x, start_y = polyline[inset_start_idx - 1]
            next_x, next_y = polyline[inset_start_idx]
            ax.arrow(
                start_x, start_y,
                next_x - start_x, next_y - start_y,
                color='cyan', head_width=arrow_size, head_length=arrow_size * 1.5,
                length_includes_head=True, zorder=2
            )

            # End arrow
            end_x, end_y = polyline[inset_end_idx]
            prev_x, prev_y = polyline[inset_end_idx - 1]
            ax.arrow(
                prev_x, prev_y,
                end_x - prev_x, end_y - prev_y,
                color='red', head_width=arrow_size, head_length=arrow_size * 1.5,
                length_includes_head=True, zorder=2
            )

    # Save or display the plot
    if save_image:
        # Save the plot as an image with exact dimensions
        fig.set_size_inches(med_image.shape[1] / dpi, med_image.shape[0] / dpi)  # Match image dimensions
        fig.savefig(output_filename, dpi=dpi, facecolor='black', transparent=True)
        plt.close(fig)
        print(f"Image saved as {output_filename} with dimensions {med_image.shape[1]}x{med_image.shape[0]}")
    else:
        plt.show()






import matplotlib.pyplot as plt
import numpy as np

def tree_map(
    root_nodes, dendrites, med_image, save_image=False, output_filename="Dendritic_Tree.tif", dpi=300
):
    """
    Visualize the tree structure with branches color-coded by level on the image.

    Parameters:
    - root_nodes: List of root TreeNode objects.
    - dendrites: Dictionary of dendrite polylines with branch IDs as keys.
    - med_image: Background image (e.g., PSD95) as a 2D numpy array.
    - save_image: Boolean flag to save the plot as a .tif file. Default is False.
    - output_filename: The filename for saving the image. Default is "Dendritic_Tree.tif".
    - dpi: Resolution in dots per inch for saving the image. Default is 300.
    """
    # Custom color sequence for levels
    colors = ['yellow', 'red', 'cyan', 'lime', 'magenta', 'blue']

    # Determine the maximum level in the tree
    max_level = 0

    # Recursive function to plot branches
    def plot_branch(node, level):
        nonlocal max_level
        max_level = max(max_level, level)
        if node.branch_id in dendrites:
            polyline = np.array(dendrites[node.branch_id])  # Ensure the polyline is a NumPy array
            ax.plot(polyline[:, 0], polyline[:, 1],
                     color=colors[level % len(colors)],
                     linewidth=1, alpha=0.8)
            # Add branch ID as the label at the midpoint of the polyline
            midpoint = len(polyline) // 2
            mid_x, mid_y = polyline[midpoint]
            ax.text(
                mid_x, mid_y, f'{node.branch_id}', fontsize=6, color='white',
                ha='center', va='center',
                bbox=dict(boxstyle="round", facecolor='black', edgecolor='none', pad=0.3)
            )
        for child in node.children:
            plot_branch(child, level + 1)

    # Compute figure size to match image dimensions
    fig, ax = plt.subplots(figsize=(med_image.shape[1] / dpi, med_image.shape[0] / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
    fig.patch.set_facecolor('black')  # Set figure background color
    ax.set_facecolor('black')        # Set axes background color
    ax.imshow(med_image, cmap='gray', interpolation='none', origin='upper', alpha=1)  # Show the PSD95 image
    ax.axis("off")  # Turn off axis lines and labels

    # Plot each root node and its subtree
    for root in root_nodes:
        plot_branch(root, level=0)

    # Add a dynamic legend
    handles = [
        plt.Line2D([0], [0], color=colors[i % len(colors)], lw=1, label=f"Level {i}")
        for i in range(max_level + 1)
    ]
    legend = ax.legend(handles=handles, loc='upper right', fontsize='small', title="Dendrite Levels", frameon=True)
    legend.get_title().set_color('white')  # Set legend title color to white
    for text in legend.get_texts():
        text.set_color('white')  # Set legend text color to white
    legend.get_frame().set_facecolor('black')  # Set legend background to black
    legend.get_frame().set_edgecolor('white')  # Set legend border color to white

    # Save or display the image
    if save_image:
        # Save the plot as an image with exact dimensions
        fig.set_size_inches(med_image.shape[1] / dpi, med_image.shape[0] / dpi)  # Match image dimensions
        fig.savefig(output_filename, dpi=dpi, facecolor='black', transparent=True)
        plt.close(fig)
        print(f"Image saved as {output_filename} with dimensions {med_image.shape[1]}x{med_image.shape[0]}")
    else:
        plt.show()
        
        
        

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours, label
import random

def mask_overlay(image, mask, mode="solid", dpi=300, save_image=False, output_filename="puncta_overlay.tif"):
    """
    Generate a 2D overlay of a 2D or 3D binary mask over a grayscale image.

    Parameters:
        image (np.ndarray): 2D grayscale image for background.
        mask (np.ndarray): 2D or 3D binary mask (Y x X or Z x Y x X).
        mode (str): "solid" for filled masks or "contour" for outlines.
        dpi (int): Display/save DPI.
        save_image (bool): Whether to save the result.
        output_filename (str): Path for saving if save_image is True.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be 2D.")
    if mask.ndim not in [2, 3]:
        raise ValueError("Mask must be either 2D or 3D.")

    # Normalize image
    image_norm = (image - np.min(image)) / (np.max(image) - np.min(image))
    image_rgb = np.dstack([image_norm] * 3)

    # Collect all overlays
    overlay_mask = np.zeros((*image.shape, 3), dtype=float)

    # Unified logic for 2D and 3D
    slices = [mask] if mask.ndim == 2 else [mask[z] for z in range(mask.shape[0])]

    def generate_random_color():
        return [random.random(), random.random(), random.random()]

    for slice_mask in slices:
        if np.sum(slice_mask) == 0:
            continue

        if mode == "solid":
            labeled = label(slice_mask)
            for label_id in range(1, labeled.max() + 1):
                color = generate_random_color()
                region = labeled == label_id
                for c in range(3):
                    overlay_mask[..., c][region] = color[c]

        elif mode == "contour":
            contours = find_contours(slice_mask.astype(float), level=0.5)
            for contour in contours:
                rr, cc = contour[:, 0].astype(int), contour[:, 1].astype(int)
                color = generate_random_color()
                valid = (rr >= 0) & (rr < image.shape[0]) & (cc >= 0) & (cc < image.shape[1])
                for c in range(3):
                    overlay_mask[rr[valid], cc[valid], c] = color[c]
        else:
            raise ValueError("mode must be either 'solid' or 'contour'")

    # Combine image and overlay (overlay replaces grayscale in colored areas)
    composite = image_rgb.copy()
    mask_any = np.any(overlay_mask > 0, axis=2)
    composite[mask_any] = overlay_mask[mask_any]

    # Plot
    fig, ax = plt.subplots(figsize=(image.shape[1] / dpi, image.shape[0] / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis("off")
    ax.imshow(composite, interpolation="none")

    if save_image:
        fig.savefig(output_filename, dpi=dpi, facecolor="black")
        plt.close(fig)
        print(f"Overlay saved as {output_filename}")
    else:
        plt.show()



def thickness_image(
    mito_mask, 
    distance_map, 
    save_image=False, 
    output_filename="distance_to_skeleton.tif"
):
    """
    Save a 32-bit float image containing actual distance-to-skeleton values 
    (thickness map) within the mitochondrial mask. Pixels outside the mitochondria are zero.

    Parameters:
        mito_mask (numpy.ndarray): Binary mask of mitochondria (2D).
        distance_map (numpy.ndarray): Distance-to-skeleton map from `distance_to_skeleton`.
        save_image (bool): Whether to save the image as a 32-bit TIFF. Default is False.
        output_filename (str): Filename for saving the image. Default is "distance_to_skeleton.tif".
    """
    import numpy as np
    import tifffile

    # Mask the distance map so only values within mitochondria are retained
    thickness_map = np.where(mito_mask, distance_map, 0).astype(np.float32)

    if save_image:
        tifffile.imwrite(output_filename, thickness_map, dtype=np.float32)
        print(f"32-bit thickness map saved as {output_filename}")
    else:
        print("Image not saved. `save_image` is set to False.")

    return thickness_map



    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from skimage.draw import polygon
from skimage.measure import find_contours
from shapely.geometry import Polygon
from scipy.spatial import cKDTree

def mito_under_synapse_nonoverlapping(
    dataframe,
    Mito_mask,
    min_distance=20,
    save_image=False,
    output_filename="puncta_visualization.tif",
    dpi=300
):
    def rectangles_overlap(poly1_coords, poly2_coords):
        poly1 = Polygon(poly1_coords)
        poly2 = Polygon(poly2_coords)
        return poly1.intersects(poly2)

    # Filter PSD95 puncta based on min distance
    positions = dataframe[["psd95_x", "psd95_y"]].values
    selected_idx = []
    kd_tree = cKDTree(positions)

    for i, pos in enumerate(positions):
        if all(np.linalg.norm(pos - positions[j]) >= min_distance for j in selected_idx):
            selected_idx.append(i)

    selected_df = dataframe.iloc[selected_idx].reset_index(drop=True)

    # Select non-overlapping rectangles
    selected_rows = []
    used_rectangles = []

    for _, row in selected_df.iterrows():
        if "Rect_coords" not in row or not isinstance(row["Rect_coords"], list):
            continue
        rect_coords = np.array(row["Rect_coords"])
        if any(rectangles_overlap(rect_coords, used) for used in used_rectangles):
            continue
        selected_rows.append(row)
        used_rectangles.append(rect_coords)

    if len(selected_rows) == 0:
        print("No non-overlapping rectangles found that meet the distance criteria.")
        return

    # Create a combined mask of all mito pixels inside rectangles
    mito_inside_rects = np.zeros_like(Mito_mask, dtype=bool)

    for row in selected_rows:
        rect_coords = np.array(row["Rect_coords"])
        rr, cc = polygon(rect_coords[:, 1], rect_coords[:, 0], shape=Mito_mask.shape)
        rect_mask = np.zeros_like(Mito_mask, dtype=bool)
        rect_mask[rr, cc] = True
        mito_inside_rects |= rect_mask & (Mito_mask > 0)

    # Find contours for visualization
    mito_contours = find_contours(mito_inside_rects.astype(float), level=0.5)

    # Plot output
    fig, ax = plt.subplots(figsize=(Mito_mask.shape[1] / dpi, Mito_mask.shape[0] / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    ax.set_xlim(0, Mito_mask.shape[1])
    ax.set_ylim(Mito_mask.shape[0], 0)
    ax.axis("off")

    # Draw lime contours
    for contour in mito_contours:
        ax.plot(contour[:, 1], contour[:, 0], color="#00FFFF", linewidth=0.5)

    # Draw rectangles and dots
    for row in selected_rows:
        x, y = row["psd95_x"], row["psd95_y"]
        rect_coords = np.array(row["Rect_coords"])
        ax.plot(x, y, 'o', color='red', markersize=1.5, markeredgewidth=0)
        ax.plot(np.append(rect_coords[:, 0], rect_coords[0, 0]),
                np.append(rect_coords[:, 1], rect_coords[0, 1]),
                color="white", linewidth=0.2, alpha=1)

    # Save or show
    if save_image:
        fig.set_size_inches(Mito_mask.shape[1] / dpi, Mito_mask.shape[0] / dpi)
        fig.savefig(output_filename, dpi=dpi, facecolor='black', transparent=True)
        plt.close(fig)
        print(f"Image saved as {output_filename}")
    else:
        plt.show()




import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from skimage.draw import polygon
from scipy.ndimage import gaussian_filter1d

def branchwise_density_map(
    df,
    polylines,
    image_shape,
    density_key="mito_area_density",  # NEW: Selectable density
    root_nodes=None,
    brightness=1.0,
    colormap_name="viridis",
    thickness=10,
    smooth_sigma=None,
    save_image=False,
    output_filename="density_map.tif",
    dpi=300,
    exclude_start_bins=5,
    exclude_end_bins=5,
):
    """
    Visualize a continuous density map for dendrites using a selected profile column.
    Compatible with the output of calculate_density_profiles (DataFrame).

    Parameters:
        df (pd.DataFrame): Output from calculate_density_profiles.
        polylines (dict): Interpolated dendrites.
        image_shape (tuple): (height, width) of image.
        density_key (str): Column from df to plot (e.g., 'mito_area_density').
    """
    density_image = np.zeros((*image_shape, 3), dtype=np.float32)
    colormap = colormaps[colormap_name]
    root_branch_ids = {n.branch_id for n in root_nodes} if root_nodes else set()

    for branch_id, group in df.groupby("branch_id"):
        if branch_id not in polylines:
            continue

        polyline = np.array(polylines[branch_id])
        if len(polyline) < 2:
            continue

        num_segments = len(polyline) - 1
        arclength = group["distance_to_soma"].values
        density = group[density_key].values.copy()

        # Exclude start/end bins
        if branch_id in root_branch_ids:
            density[:exclude_start_bins] = 0
        else:
            density[-exclude_end_bins:] = 0

        # Smooth if requested
        if smooth_sigma is not None:
            density = gaussian_filter1d(density, sigma=smooth_sigma)

        # Interpolate density to number of segments
        interpolated_density = np.interp(
            np.linspace(0, arclength[-1], num_segments),
            arclength,
            density
        )

        # Normalize and apply brightness
        max_density = interpolated_density.max() if interpolated_density.max() > 0 else 1
        normalized_density = interpolated_density / max_density
        interpolated_colors = colormap(normalized_density)[:, :3] * brightness
        interpolated_colors = np.clip(interpolated_colors, 0, 1)

        for i in range(num_segments):
            x0, y0 = polyline[i]
            x1, y1 = polyline[i + 1]

            dx, dy = x1 - x0, y1 - y0
            length = np.hypot(dx, dy)
            if length == 0:
                continue

            perp_dx = -dy / length
            perp_dy = dx / length
            half_thickness = thickness / 2

            vertices = np.array([
                [x0 + perp_dx * half_thickness, y0 + perp_dy * half_thickness],
                [x0 - perp_dx * half_thickness, y0 - perp_dy * half_thickness],
                [x1 - perp_dx * half_thickness, y1 - perp_dy * half_thickness],
                [x1 + perp_dx * half_thickness, y1 + perp_dy * half_thickness]
            ])

            rr, cc = polygon(vertices[:, 1], vertices[:, 0], shape=image_shape)
            density_image[rr, cc] += interpolated_colors[i]

    density_image = np.clip(density_image, 0, 1)

    def setup_and_display_figure():
        fig, ax = plt.subplots(figsize=(image_shape[1] / dpi, image_shape[0] / dpi), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.patch.set_facecolor("black")
        ax.axis("off")
        ax.imshow(density_image, interpolation="none")
        return fig, ax

    if save_image:
        fig, ax = setup_and_display_figure()
        fig.savefig(output_filename, dpi=dpi, facecolor="black")
        plt.close(fig)
        print(f"Density map saved as {output_filename}")
    else:
        fig, ax = setup_and_display_figure()
        plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from skimage.draw import polygon
from scipy.ndimage import gaussian_filter1d

def global_density_map(
    df,
    polylines,
    image_shape,
    density_key="mito_area_density",
    root_nodes=None,
    brightness=1.0,
    colormap_name="viridis",
    thickness=5,
    save_image=False,
    output_filename="global_density_map.tif",
    dpi=300,
    exclude_start_bins=10,
    exclude_end_bins=10,
):
    """
    Visualize a global normalized continuous density map for dendrites using the selected profile column.

    Parameters:
        df (pd.DataFrame): Output from calculate_density_profiles.
        polylines (dict): Interpolated dendrites with branch_id as keys.
        image_shape (tuple): (height, width) of output image.
        density_key (str): Column from df to visualize.
        root_nodes (list): List of TreeNode objects (optional).
        brightness (float): Scale factor for color brightness.
        colormap_name (str): Matplotlib colormap name.
        thickness (int): Polyline thickness.
        save_image (bool): Whether to save the output.
        output_filename (str): Filename to save the image.
        dpi (int): DPI for saving.
        exclude_start_bins (int): Bins to zero out at start of root branches.
        exclude_end_bins (int): Bins to zero out at end of non-root branches.
    """
    density_image = np.zeros((*image_shape, 3), dtype=np.float32)
    colormap = colormaps[colormap_name]
    root_branch_ids = {n.branch_id for n in root_nodes} if root_nodes else set()

    # Zero-out ends and find global max
    df_copy = df.copy()
    for branch_id in df_copy["branch_id"].unique():
        branch_mask = df_copy["branch_id"] == branch_id
        branch_df = df_copy[branch_mask]
        n = len(branch_df)
        if n < 2:
            continue
        if branch_id in root_branch_ids:
            df_copy.loc[branch_mask, density_key].iloc[:exclude_start_bins] = 0
        else:
            df_copy.loc[branch_mask, density_key].iloc[-exclude_end_bins:] = 0

    global_max = df_copy[density_key].max()
    if global_max == 0:
        global_max = 1

    for branch_id, group in df_copy.groupby("branch_id"):
        if branch_id not in polylines:
            continue
        polyline = np.array(polylines[branch_id])
        if len(polyline) < 2:
            continue

        num_segments = len(polyline) - 1
        arclength = group["distance_to_soma"].values
        density = group[density_key].values

        # Interpolate density to match number of segments
        interpolated_density = np.interp(
            np.linspace(0, arclength[-1], num_segments),
            arclength,
            density
        )

        # Global normalization
        normalized_density = interpolated_density / global_max
        interpolated_colors = colormap(normalized_density)[:, :3] * brightness
        interpolated_colors = np.clip(interpolated_colors, 0, 1)

        for i in range(num_segments):
            x0, y0 = polyline[i]
            x1, y1 = polyline[i + 1]

            dx, dy = x1 - x0, y1 - y0
            seg_len = np.hypot(dx, dy)
            if seg_len == 0:
                continue

            perp_dx = -dy / seg_len
            perp_dy = dx / seg_len
            half_thick = thickness / 2

            vertices = np.array([
                [x0 + perp_dx * half_thick, y0 + perp_dy * half_thick],
                [x0 - perp_dx * half_thick, y0 - perp_dy * half_thick],
                [x1 - perp_dx * half_thick, y1 - perp_dy * half_thick],
                [x1 + perp_dx * half_thick, y1 + perp_dy * half_thick]
            ])
            rr, cc = polygon(vertices[:, 1], vertices[:, 0], shape=image_shape)
            density_image[rr, cc] += interpolated_colors[i]

    density_image = np.clip(density_image, 0, 1)

    def setup_and_display_figure():
        fig, ax = plt.subplots(figsize=(image_shape[1] / dpi, image_shape[0] / dpi), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.patch.set_facecolor("black")
        ax.axis("off")
        ax.imshow(density_image, interpolation="none")
        return fig, ax

    if save_image:
        fig, ax = setup_and_display_figure()
        fig.savefig(output_filename, dpi=dpi, facecolor="black")
        plt.close(fig)
        print(f"Global density map saved as {output_filename}")
    else:
        fig, ax = setup_and_display_figure()
        plt.show()



import pandas as pd
import numpy as np

class TreeNode:
    def __init__(self, branch_id):
        self.branch_id = branch_id
        self.children = []
        self.level = None
        self.parent = None

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)


def make_tree(csv_path, prune=False):
    """
    Create a hierarchical tree structure from dendritic data in a CSV file.

    Parameters:
        csv_path (str): Path to the CSV file containing columns Branch_ID, Node_ID, X, Y.
        prune (bool): Whether to prune non-level 0 polylines by replacing their first node
                      with the exact intersection point with their parent. Default is False.

    Returns:
        list: List of root TreeNode objects representing the hierarchical tree.
        dict: Dictionary of pruned polylines (if prune is True).
    """
    # Helper functions
    def create_polylines(csv_path):
        df = pd.read_csv(csv_path)
        required_columns = {"Branch_ID", "Node_ID", "X", "Y"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")
        df_sorted = df.sort_values(by=["Branch_ID", "Node_ID"])
        return {
            branch_id: np.array(list(zip(group["X"], group["Y"])))
            for branch_id, group in df_sorted.groupby("Branch_ID")
        }

    def exact_intersection_point(p1, p2, q1, q2):
        p1, p2, q1, q2 = map(np.array, (p1, p2, q1, q2))
        dp = p2 - p1
        dq = q2 - q1
        denom = dq[0] * dp[1] - dq[1] * dp[0]
        if abs(denom) < 1e-10:
            return None
        t = ((q1[0] - p1[0]) * (q1[1] - q2[1]) - (q1[1] - p1[1]) * (q1[0] - q2[0])) / denom
        u = ((q1[0] - p1[0]) * (p1[1] - p2[1]) - (q1[1] - p1[1]) * (p1[0] - p2[0])) / denom
        if 0 <= t <= 1 and 0 <= u <= 1:
            return p1 + t * dp
        return None

    def find_exact_intersection(child_polyline, parent_polyline):
        for i in range(min(len(child_polyline) - 1, 3)):  # Limit to nodes 0-3 of the child
            for j in range(len(parent_polyline) - 1):  # Iterate over all parent segments
                intersection = exact_intersection_point(
                    child_polyline[i], child_polyline[i + 1],
                    parent_polyline[j], parent_polyline[j + 1]
                )
                if intersection is not None:
                    return intersection
        return None

    def assign_levels(polylines, prune):
        nodes = {branch_id: TreeNode(branch_id) for branch_id in polylines.keys()}
        unassigned = set(polylines.keys())
        pruned_polylines = polylines.copy()

        # Assign level 0 roots
        root_nodes = []
        for branch_id1 in list(unassigned):
            is_root = all(
                find_exact_intersection(polylines[branch_id1], polylines[branch_id2]) is None
                for branch_id2 in polylines if branch_id1 != branch_id2
            )
            if is_root:
                nodes[branch_id1].level = 0
                root_nodes.append(nodes[branch_id1])
                unassigned.remove(branch_id1)

        # Assign higher levels iteratively
        current_level = 1
        current_nodes = root_nodes
        while unassigned:
            assigned_this_level = set()
            next_level_nodes = []
            for branch_id1 in list(unassigned):
                for parent_node in current_nodes:  # Traverse current level's nodes
                    if parent_node.level == current_level - 1:
                        intersection = find_exact_intersection(polylines[branch_id1], polylines[parent_node.branch_id])
                        if intersection is not None:
                            if prune:
                                pruned_polylines[branch_id1][0] = intersection
                            parent_node.add_child(nodes[branch_id1])
                            nodes[branch_id1].level = current_level
                            assigned_this_level.add(branch_id1)
                            next_level_nodes.append(nodes[branch_id1])
                            break
            unassigned -= assigned_this_level
            current_nodes = next_level_nodes  # Move to the next level's nodes
            if not assigned_this_level:
                break
            current_level += 1

        return root_nodes, pruned_polylines

    def display_tree(node, level=0):
        print("  " * level + f"Branch {node.branch_id} (Level {node.level})")
        for child in node.children:
            display_tree(child, level + 1)

    # Main logic
    polylines = create_polylines(csv_path)
    root_nodes, pruned_polylines = assign_levels(polylines, prune)

    for root in root_nodes:
        display_tree(root)

    return root_nodes, pruned_polylines if prune else None


import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from skimage.feature import blob_log
from skimage.draw import disk
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


def detect_puncta(
    PSD95_image, 
    root_nodes, 
    dendrites, 
    radius_dendrite=50,
    min_sigma=1.0, 
    max_sigma=1.5, 
    num_sigma=10, 
    threshold_factor=1, 
    min_size=5,
    step=1
):
    if PSD95_image.ndim != 2:
        raise ValueError("Input PSD95_image must be a 2D image.")

    PSD95_mask_stack = []
    all_properties = []

    def cumulative_distances(polyline):
        return np.cumsum([0] + [np.linalg.norm(polyline[i+1] - polyline[i]) for i in range(len(polyline) - 1)])

    def interpolate_polyline(polyline):
        if len(polyline) < 2:
            return polyline
        distances = cumulative_distances(polyline)
        interpolator = interp1d(distances, polyline, axis=0, kind="linear", assume_sorted=True)
        dense_distances = np.arange(0, distances[-1], step)
        return interpolator(dense_distances)

    def build_kd_tree_and_map():
        interpolated = {bid: interpolate_polyline(p) for bid, p in dendrites.items()}
        points = np.vstack(list(interpolated.values()))
        tree = cKDTree(points)
        return tree, points, interpolated

    def calculate_branch_length(polyline):
        return np.sum([np.linalg.norm(polyline[i+1] - polyline[i]) for i in range(len(polyline) - 1)])

    def trace_to_soma(branch_id, point_idx, interpolated):
        def collect_tree_nodes(node):
            return [node] + sum([collect_tree_nodes(child) for child in node.children], [])
        node_map = {n.branch_id: n for root in root_nodes for n in collect_tree_nodes(root)}
        polyline = interpolated[branch_id]
        distance = cumulative_distances(polyline)[point_idx]
        node = node_map.get(branch_id, None)
        level = 0
        while node and getattr(node, "parent", None):
            parent = node.parent
            parent_poly = interpolated[parent.branch_id]
            parent_dist = cumulative_distances(parent_poly)
            join_idx = np.argmin(np.linalg.norm(parent_poly - polyline[0], axis=1))
            distance += parent_dist[join_idx]
            polyline = parent_poly
            node = parent
            level += 1
        return distance, level

    kd_tree, tree_points, interpolated_polylines = build_kd_tree_and_map()

    def find_nearest_branch_and_index(centroid):
        dist, idx = kd_tree.query(centroid.reshape(1, -1), k=1)
        closest_point = tree_points[idx[0]]
        for branch_id, polyline in interpolated_polylines.items():
            match_idx = np.where(np.all(np.isclose(polyline, closest_point, atol=1e-3), axis=1))[0]
            if len(match_idx) > 0:
                return branch_id, match_idx[0]
        return None, None

    def get_pixels_near_branches(branch_ids):
        mask = np.zeros_like(PSD95_image, dtype=bool)
        for branch_id in branch_ids:
            if branch_id in dendrites:
                for point in dendrites[branch_id]:
                    rr, cc = disk((int(point[1]), int(point[0])), radius_dendrite, shape=PSD95_image.shape)
                    mask[rr, cc] = True
        return mask

    def collect_all_branch_ids(node):
        branch_ids = [node.branch_id]
        for child in node.children:
            branch_ids.extend(collect_all_branch_ids(child))
        return branch_ids

    for root_node in root_nodes:
        branch_ids = collect_all_branch_ids(root_node)
        pixels_near_branches = PSD95_image[get_pixels_near_branches(branch_ids)]
        mean_intensity = np.mean(pixels_near_branches)
        std_intensity = np.std(pixels_near_branches)
        threshold = mean_intensity + threshold_factor * std_intensity
        dendrite_mask = get_pixels_near_branches(branch_ids)

        blobs = blob_log(PSD95_image, min_sigma=min_sigma, max_sigma=max_sigma,
                         num_sigma=num_sigma, threshold=threshold)

        for y, x, sigma in blobs:
            y, x = int(y), int(x)
            if not dendrite_mask[y, x]:
                continue
            radius_otsu = int(np.ceil(3 * sigma))
            rr, cc = disk((y, x), radius_otsu, shape=PSD95_image.shape)
            local_mask = np.zeros_like(PSD95_image, dtype=bool)
            local_mask[rr, cc] = True

            try:
                local_vals = PSD95_image[rr, cc]
                otsu_thresh = threshold_otsu(local_vals)
            except ValueError:
                continue

            binary_local = (PSD95_image > otsu_thresh) & local_mask
            if np.sum(binary_local) == 0:
                continue
            labeled = label(binary_local)
            props = regionprops(labeled)
            if not props:
                continue
            largest = max(props, key=lambda r: r.area)
            if largest.area < min_size:
                continue
            region_mask = (labeled == largest.label)
            centroid = np.array(largest.centroid)[::-1]

            branch_id, idx = find_nearest_branch_and_index(centroid)
            if branch_id is None or idx is None:
                continue
            distance_to_soma_px, branch_level = trace_to_soma(branch_id, idx, interpolated_polylines)
            branch_length_px = calculate_branch_length(np.array(dendrites[branch_id]))

            placed = False
            for z in range(len(PSD95_mask_stack)):
                if not np.any(PSD95_mask_stack[z] & region_mask):
                    PSD95_mask_stack[z][region_mask] = True
                    all_properties.append({
                        "z": z,
                        "centroid": largest.centroid,
                        "area": largest.area,
                        "perimeter": largest.perimeter,
                        "distance_to_soma_px": distance_to_soma_px,
                        "branch_id": branch_id,
                        "branch_level": branch_level,
                        "branch_length_px": branch_length_px
                    })
                    placed = True
                    break

            if not placed:
                new_slice = np.zeros_like(PSD95_image, dtype=bool)
                new_slice[region_mask] = True
                PSD95_mask_stack.append(new_slice)
                all_properties.append({
                    "z": len(PSD95_mask_stack) - 1,
                    "centroid": largest.centroid,
                    "area": largest.area,
                    "perimeter": largest.perimeter,
                    "distance_to_soma_px": distance_to_soma_px,
                    "branch_id": branch_id,
                    "branch_level": branch_level,
                    "branch_length_px": branch_length_px
                })

    PSD95_mask_3d = np.stack(PSD95_mask_stack, axis=0) if PSD95_mask_stack else np.zeros((1, *PSD95_image.shape), dtype=bool)
    return all_properties, PSD95_mask_3d



def detect_puncta2(
    PSD95_image, 
    root_nodes, 
    dendrites, 
    radius_dendrite=50,
    min_sigma=1.0, 
    max_sigma=1.5, 
    num_sigma=10, 
    threshold_factor=1, 
    min_size=5,
    step=1,
    min_distance=0,
    max_distance=np.inf
):
    if PSD95_image.ndim != 2:
        raise ValueError("Input PSD95_image must be a 2D image.")

    PSD95_mask_stack = []
    all_properties = []

    def cumulative_distances(polyline):
        return np.cumsum([0] + [np.linalg.norm(polyline[i+1] - polyline[i]) for i in range(len(polyline) - 1)])

    def interpolate_polyline(polyline):
        if len(polyline) < 2:
            return polyline
        distances = cumulative_distances(polyline)
        from scipy.interpolate import interp1d
        interpolator = interp1d(distances, polyline, axis=0, kind="linear", assume_sorted=True)
        dense_distances = np.arange(0, distances[-1], step)
        return interpolator(dense_distances)

    def build_kd_tree_and_map():
        from scipy.spatial import cKDTree
        interpolated = {bid: interpolate_polyline(p) for bid, p in dendrites.items()}
        points = np.vstack(list(interpolated.values()))
        tree = cKDTree(points)
        return tree, points, interpolated

    def calculate_branch_length(polyline):
        return np.sum([np.linalg.norm(polyline[i+1] - polyline[i]) for i in range(len(polyline) - 1)])

    def trace_to_soma(branch_id, point_idx, interpolated):
        def collect_tree_nodes(node):
            return [node] + sum([collect_tree_nodes(child) for child in node.children], [])
        node_map = {n.branch_id: n for root in root_nodes for n in collect_tree_nodes(root)}
        polyline = interpolated[branch_id]
        distance = cumulative_distances(polyline)[point_idx]
        node = node_map.get(branch_id, None)
        level = 0
        while node and getattr(node, "parent", None):
            parent = node.parent
            parent_poly = interpolated[parent.branch_id]
            parent_dist = cumulative_distances(parent_poly)
            join_idx = np.argmin(np.linalg.norm(parent_poly - polyline[0], axis=1))
            distance += parent_dist[join_idx]
            polyline = parent_poly
            node = parent
            level += 1
        return distance, level

    kd_tree, tree_points, interpolated_polylines = build_kd_tree_and_map()

    def find_nearest_branch_and_index(centroid):
        dist, idx = kd_tree.query(centroid.reshape(1, -1), k=1)
        closest_point = tree_points[idx[0]]
        for branch_id, polyline in interpolated_polylines.items():
            match_idx = np.where(np.all(np.isclose(polyline, closest_point, atol=1e-3), axis=1))[0]
            if len(match_idx) > 0:
                return branch_id, match_idx[0]
        return None, None

    def get_pixels_near_branches(branch_ids):
        from skimage.draw import disk
        mask = np.zeros_like(PSD95_image, dtype=bool)
        for branch_id in branch_ids:
            if branch_id in dendrites:
                for point in dendrites[branch_id]:
                    rr, cc = disk((int(point[1]), int(point[0])), radius_dendrite, shape=PSD95_image.shape)
                    mask[rr, cc] = True
        return mask

    def collect_all_branch_ids(node):
        branch_ids = [node.branch_id]
        for child in node.children:
            branch_ids.extend(collect_all_branch_ids(child))
        return branch_ids

    for root_node in root_nodes:
        branch_ids = collect_all_branch_ids(root_node)
        pixels_near_branches = PSD95_image[get_pixels_near_branches(branch_ids)]
        mean_intensity = np.mean(pixels_near_branches)
        std_intensity = np.std(pixels_near_branches)
        threshold = mean_intensity + threshold_factor * std_intensity
        dendrite_mask = get_pixels_near_branches(branch_ids)

        from skimage.feature import blob_log
        blobs = blob_log(PSD95_image, min_sigma=min_sigma, max_sigma=max_sigma,
                         num_sigma=num_sigma, threshold=threshold)

        for y, x, sigma in blobs:
            y, x = int(y), int(x)
            if not dendrite_mask[y, x]:
                continue
            radius_otsu = int(np.ceil(3 * sigma))
            from skimage.draw import disk
            rr, cc = disk((y, x), radius_otsu, shape=PSD95_image.shape)
            local_mask = np.zeros_like(PSD95_image, dtype=bool)
            local_mask[rr, cc] = True

            try:
                local_vals = PSD95_image[rr, cc]
                from skimage.filters import threshold_otsu
                otsu_thresh = threshold_otsu(local_vals)
            except ValueError:
                continue

            binary_local = (PSD95_image > otsu_thresh) & local_mask
            if np.sum(binary_local) == 0:
                continue
            from skimage.measure import regionprops, label
            labeled = label(binary_local)
            props = regionprops(labeled)
            if not props:
                continue
            largest = max(props, key=lambda r: r.area)
            if largest.area < min_size:
                continue
            region_mask = (labeled == largest.label)
            centroid = np.array(largest.centroid)[::-1]

            # --- Compute distance to nearest dendrite point ---
            dist, idx = kd_tree.query(centroid.reshape(1, -1), k=1)
            if not (min_distance <= dist[0] <= max_distance):
                continue

            branch_id, idx2 = find_nearest_branch_and_index(centroid)
            if branch_id is None or idx2 is None:
                continue
            distance_to_soma_px, branch_level = trace_to_soma(branch_id, idx2, interpolated_polylines)
            branch_length_px = calculate_branch_length(np.array(dendrites[branch_id]))

            placed = False
            for z in range(len(PSD95_mask_stack)):
                if not np.any(PSD95_mask_stack[z] & region_mask):
                    PSD95_mask_stack[z][region_mask] = True
                    all_properties.append({
                        "z": z,
                        "centroid": largest.centroid,
                        "area": largest.area,
                        "perimeter": largest.perimeter,
                        "distance_to_soma_px": distance_to_soma_px,
                        "branch_id": branch_id,
                        "branch_level": branch_level,
                        "branch_length_px": branch_length_px
                    })
                    placed = True
                    break

            if not placed:
                new_slice = np.zeros_like(PSD95_image, dtype=bool)
                new_slice[region_mask] = True
                PSD95_mask_stack.append(new_slice)
                all_properties.append({
                    "z": len(PSD95_mask_stack) - 1,
                    "centroid": largest.centroid,
                    "area": largest.area,
                    "perimeter": largest.perimeter,
                    "distance_to_soma_px": distance_to_soma_px,
                    "branch_id": branch_id,
                    "branch_level": branch_level,
                    "branch_length_px": branch_length_px
                })

    PSD95_mask_3d = np.stack(PSD95_mask_stack, axis=0) if PSD95_mask_stack else np.zeros((1, *PSD95_image.shape), dtype=bool)
    return all_properties, PSD95_mask_3d








from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.draw import disk
import numpy as np

def detect_objects(
    Mito_image,
    root_nodes,
    dendrites,
    radius=25,
    bin_size=100,
    overlap_radius=25,
    min_overlap=0.8,
    min_size=50
):
    """
    Perform Otsu thresholding with a sliding window of bin size along the dendritic polyline.
    Filter objects based on overlap with dendritic regions and size.

    Parameters:
        Mito_image (numpy.ndarray): 2D mitochondrial image.
        root_nodes (list): List of root TreeNode objects representing the hierarchical dendritic tree.
        dendrites (dict): Dictionary of Branch_ID to polyline coordinates.
        radius (int): Radius around each polyline point for defining the window width.
        bin_size (int): Number of polyline points to include in each window (sliding bin).
        overlap_radius (int): Radius for checking object overlap with dendrites. Default is 25.
        min_overlap (float): Minimum fraction of overlap for an object to be retained. Default is 0.8.
        min_size (int): Minimum size (in pixels) for an object to be retained. Default is 50.

    Returns:
        numpy.ndarray: Final binary mask of filtered objects.
    """
    def get_all_branch_ids(node):
        """Recursively collect all branch IDs in the subtree rooted at the given node."""
        branch_ids = [node.branch_id]
        for child in node.children:
            branch_ids.extend(get_all_branch_ids(child))
        return branch_ids

    def get_mask_for_branches(branch_ids, radius):
        """Create a mask for regions near the given branches within the specified radius."""
        mask = np.zeros_like(Mito_image, dtype=bool)
        for branch_id in branch_ids:
            if branch_id in dendrites:
                polyline = dendrites[branch_id]
                for point in polyline:
                    rr, cc = disk((int(point[1]), int(point[0])), radius, shape=Mito_image.shape)
                    mask[rr, cc] = True
        return mask

    # Initialize the Otsu mask
    otsu_mask = np.zeros_like(Mito_image, dtype=bool)

    for branch_id, polyline in dendrites.items():
        polyline = np.array(polyline)

        for i in range(0, len(polyline), bin_size // 2):  # Overlapping bins
            # Get the current window of points
            window_points = polyline[i:i + bin_size]

            if len(window_points) < 2:  # Skip too small windows
                continue

            # Create a binary mask for this bin
            bin_mask = np.zeros_like(Mito_image, dtype=bool)
            for point in window_points:
                rr, cc = disk((int(point[1]), int(point[0])), radius, shape=Mito_image.shape)
                bin_mask[rr, cc] = True

            # Extract the local region for this bin
            local_region = Mito_image[bin_mask]

            # Apply Otsu thresholding
            if local_region.size > 0:
                local_threshold = threshold_otsu(local_region)
                otsu_mask[bin_mask] |= Mito_image[bin_mask] > local_threshold

    # Filter objects based on overlap with dendritic regions and size
    final_mask = np.zeros_like(otsu_mask, dtype=bool)
    labeled_mask = label(otsu_mask)
    branch_overlap_mask = get_mask_for_branches(list(dendrites.keys()), overlap_radius)

    for prop in regionprops(labeled_mask):
        object_pixels = labeled_mask == prop.label
        overlap_pixels = np.sum(object_pixels & branch_overlap_mask)
        total_pixels = np.sum(object_pixels)

        # Retain the object if it satisfies the overlap and size criteria
        if total_pixels >= min_size and (overlap_pixels / total_pixels) >= min_overlap:
            final_mask[object_pixels] = True

    return final_mask
    
'''
from skimage.morphology import skeletonize, binary_erosion
from skimage.measure import label, regionprops
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd

def remodeled_mito_width(
    mito_mask, std_multiplier=1.0, longest=False, skeleton_threshold=10,
    num_nearest=5, min_region_size=0
):
    """
    Measure local thickness for mitochondrial objects by mapping pixels to skeleton points,
    assigning the average of distances from the N nearest border pixels, and normalizing
    thickness values per mitochondrion (0–1 scaling within each object).

    Parameters:
        mito_mask (numpy.ndarray): Binary mask of mitochondria (2D).
        std_multiplier (float): Multiplier for standard deviation in thresholding.
        longest (bool): If True, retain only the skeleton along its longest path for each mitochondrial object.
        skeleton_threshold (int): Minimum number of skeleton pixels required to process the object.
        num_nearest (int): Number of nearest border pixels to consider for distance averaging.
        min_region_size (int): Minimum number of pixels required for a region to be retained. Default is 0 (keep all).

    Returns:
        tuple:
            - numpy.ndarray: Per-mito normalized thickness map.
            - numpy.ndarray: Thresholded thickness map (not normalized).
            - pandas.DataFrame: Region statistics for each mitochondrial object and connected region.
    """
    # Label the mito objects
    labeled_mito = label(mito_mask, connectivity=2)
    num_objects = labeled_mito.max()
    thickness_map = np.zeros_like(mito_mask, dtype=np.float32)
    raw_thickness_map = np.zeros_like(mito_mask, dtype=np.float32)  # For unnormalized values
    thresholded_map = np.zeros_like(mito_mask, dtype=np.float32)
    region_stats = []

    for obj_id in range(1, num_objects + 1):
        obj_mask = (labeled_mito == obj_id)
        skeleton = skeletonize(obj_mask)

        skeleton_points = np.column_stack(np.where(skeleton))
        if len(skeleton_points) < skeleton_threshold:
            continue

        if longest:
            skeleton = extract_longest_path_skeleton(skeleton)

        skeleton_points = np.column_stack(np.where(skeleton))
        if len(skeleton_points) == 0:
            continue

        object_points = np.column_stack(np.where(obj_mask))
        skeleton_tree = cKDTree(skeleton_points)
        distances, indices = skeleton_tree.query(object_points)

        skeleton_to_pixels = {tuple(sp): [] for sp in skeleton_points}
        for pixel, sk_idx in zip(object_points, indices):
            skeleton_to_pixels[tuple(skeleton_points[sk_idx])].append(tuple(pixel))

        border_mask = (obj_mask & ~binary_erosion(obj_mask))
        border_pixels = np.column_stack(np.where(border_mask))
        if len(border_pixels) > 0:
            border_tree = cKDTree(border_pixels)

        skeleton_thickness = []
        for sk_point, mapped_pixels in skeleton_to_pixels.items():
            mapped_pixels = np.array(mapped_pixels)
            if len(border_pixels) > 0:
                distances, _ = border_tree.query(sk_point, k=min(num_nearest, len(border_pixels)))
                avg_distance = np.mean(distances)
                for pixel in mapped_pixels:
                    raw_thickness_map[pixel[0], pixel[1]] = avg_distance
                skeleton_thickness.append(avg_distance)
            else:
                for pixel in mapped_pixels:
                    raw_thickness_map[pixel[0], pixel[1]] = 0
                skeleton_thickness.append(0)

        # Normalize thickness values within the object
        obj_raw_values = raw_thickness_map[obj_mask]
        max_val = obj_raw_values.max()
        min_val = obj_raw_values.min()
        if max_val > min_val:
            normalized_values = (obj_raw_values - min_val) / (max_val - min_val)
            thickness_map[obj_mask] = normalized_values
        else:
            thickness_map[obj_mask] = 0  # All values are equal or zero

        mean_thickness = np.mean(skeleton_thickness) if skeleton_thickness else 0
        std_thickness = np.std(skeleton_thickness) if skeleton_thickness else 0
        threshold = mean_thickness + std_multiplier * std_thickness
        obj_thresholded = np.where(raw_thickness_map[obj_mask] > threshold, raw_thickness_map[obj_mask], 0)

        temp_mask = np.zeros_like(mito_mask, dtype=np.float32)
        temp_mask[obj_mask] = obj_thresholded
        labeled_regions = label(temp_mask > 0, connectivity=2)
        region_props = regionprops(labeled_regions)

        for region in region_props:
            if region.area < min_region_size:
                continue

            region_coords = region.coords
            max_thickness = raw_thickness_map[region_coords[:, 0], region_coords[:, 1]].max()

            for coord in region_coords:
                thresholded_map[coord[0], coord[1]] = raw_thickness_map[coord[0], coord[1]]

            region_stats.append({
                "Mito_ID": obj_id,
                "Region_ID": region.label,
                "Mito_Area": np.sum(obj_mask),
                "Region_Area": region.area,
                "Mean_Thickness": mean_thickness,
                "Max_Thickness": max_thickness,
                "Centroid_X": region.centroid[1],
                "Centroid_Y": region.centroid[0]
            })

    region_stats_df = pd.DataFrame(region_stats)
    return thickness_map, thresholded_map, region_stats_df
'''    

import numpy as np
import pandas as pd
from skimage.morphology import skeletonize, binary_erosion
from skimage.measure import label, regionprops
from scipy.spatial import cKDTree

def remodeled_mito_width(
    mito_mask, root_nodes, dendrites, std_multiplier=1.0,
    skeleton_threshold=10, num_nearest=5, min_region_size=0, step=1
):
    from scipy.interpolate import interp1d

    def cumulative_distances(polyline):
        return np.cumsum([0] + [np.linalg.norm(polyline[i + 1] - polyline[i]) for i in range(len(polyline) - 1)])

    def interpolate_polyline(polyline):
        if len(polyline) < 2:
            return polyline
        distances = cumulative_distances(polyline)
        interpolator = interp1d(distances, polyline, axis=0, kind="linear", assume_sorted=True)
        dense_distances = np.arange(0, distances[-1], step)
        return interpolator(dense_distances)

    def build_kd_tree_and_map():
        interpolated = {bid: interpolate_polyline(p) for bid, p in dendrites.items()}
        points = np.vstack(list(interpolated.values()))
        tree = cKDTree(points)
        return tree, points, interpolated

    def trace_to_soma(branch_id, point_idx, interpolated):
        def collect_tree_nodes(node):
            return [node] + sum([collect_tree_nodes(child) for child in node.children], [])
        node_map = {n.branch_id: n for root in root_nodes for n in collect_tree_nodes(root)}
        polyline = interpolated[branch_id]
        distance = cumulative_distances(polyline)[point_idx]
        level = 0
        node = node_map.get(branch_id, None)
        while node and getattr(node, "parent", None):
            parent = node.parent
            parent_poly = interpolated[parent.branch_id]
            parent_dist = cumulative_distances(parent_poly)
            join_idx = np.argmin(np.linalg.norm(parent_poly - polyline[0], axis=1))
            distance += parent_dist[join_idx]
            polyline = parent_poly
            node = parent
            level += 1
        return distance, level

    def extract_longest_path_skeleton(skeleton):
        import networkx as nx
        coords = np.column_stack(np.where(skeleton))
        G = nx.Graph()
        for y, x in coords:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx_ = y + dy, x + dx
                    if 0 <= ny < skeleton.shape[0] and 0 <= nx_ < skeleton.shape[1] and skeleton[ny, nx_]:
                        dist = np.sqrt(dy**2 + dx**2)
                        G.add_edge((y, x), (ny, nx_), weight=dist)
        if len(G.nodes) == 0:
            return np.zeros_like(skeleton, dtype=bool)
        endpoints = [n for n in G.nodes if G.degree[n] == 1]
        if len(endpoints) < 2:
            return skeleton
        max_path = []
        max_length = 0
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                try:
                    path = nx.shortest_path(G, endpoints[i], endpoints[j], weight='weight')
                    path_length = sum(np.linalg.norm(np.subtract(path[k + 1], path[k])) for k in range(len(path) - 1))
                    if path_length > max_length:
                        max_length = path_length
                        max_path = path
                except nx.NetworkXNoPath:
                    continue
        longest_mask = np.zeros_like(skeleton, dtype=bool)
        for y, x in max_path:
            longest_mask[y, x] = True
        return longest_mask

    def calculate_skeleton_length(skeleton):
        coords = np.column_stack(np.where(skeleton))
        total_length = 0.0
        neighbors_offset = np.array([
            [-1, -1], [-1, 0], [-1, 1],
            [ 0, -1],          [ 0, 1],
            [ 1, -1], [ 1, 0], [ 1, 1]
        ])
        for coord in coords:
            neighbors = coord + neighbors_offset
            valid = (
                (neighbors[:, 0] >= 0) & (neighbors[:, 0] < skeleton.shape[0]) &
                (neighbors[:, 1] >= 0) & (neighbors[:, 1] < skeleton.shape[1])
            )
            valid_neighbors = neighbors[valid]
            valid_neighbors = valid_neighbors[skeleton[valid_neighbors[:, 0], valid_neighbors[:, 1]]]
            distances = np.sqrt(np.sum((valid_neighbors - coord) ** 2, axis=1))
            total_length += np.sum(distances) / 2
        return total_length

    kd_tree, tree_points, interpolated_polylines = build_kd_tree_and_map()

    labeled_mito = label(mito_mask, connectivity=2)
    thickness_map = np.zeros_like(mito_mask, dtype=np.float32)
    raw_thickness_map = np.zeros_like(mito_mask, dtype=np.float32)
    thresholded_map = np.zeros_like(mito_mask, dtype=np.float32)
    binary_skeleton_map = np.zeros_like(mito_mask, dtype=bool)
    region_stats = []

    for obj_id in range(1, labeled_mito.max() + 1):
        obj_mask = (labeled_mito == obj_id)
        skeleton = skeletonize(obj_mask)
        if np.sum(skeleton) < skeleton_threshold:
            continue

        binary_skeleton_map |= skeleton

        length_skeleton = extract_longest_path_skeleton(skeleton)
        mito_length = calculate_skeleton_length(length_skeleton)

        object_points = np.column_stack(np.where(obj_mask))
        skeleton_points = np.column_stack(np.where(skeleton))
        skeleton_tree = cKDTree(skeleton_points)
        _, indices = skeleton_tree.query(object_points)

        skeleton_to_pixels = {tuple(sp): [] for sp in skeleton_points}
        for pixel, sk_idx in zip(object_points, indices):
            skeleton_to_pixels[tuple(skeleton_points[sk_idx])].append(tuple(pixel))

        border_mask = obj_mask & ~binary_erosion(obj_mask)
        border_pixels = np.column_stack(np.where(border_mask))
        if len(border_pixels) > 0:
            border_tree = cKDTree(border_pixels)

        skeleton_thickness = []
        for sk_point, mapped_pixels in skeleton_to_pixels.items():
            mapped_pixels = np.array(mapped_pixels)
            if len(border_pixels) > 0:
                distances, _ = border_tree.query(sk_point, k=min(num_nearest, len(border_pixels)))
                avg_distance = np.mean(distances)
                for pixel in mapped_pixels:
                    raw_thickness_map[pixel[0], pixel[1]] = avg_distance
                skeleton_thickness.append(avg_distance)
            else:
                for pixel in mapped_pixels:
                    raw_thickness_map[pixel[0], pixel[1]] = 0
                skeleton_thickness.append(0)

        obj_raw_values = raw_thickness_map[obj_mask]
        max_val = obj_raw_values.max()
        min_val = obj_raw_values.min()
        thickness_map[obj_mask] = (obj_raw_values - min_val) / (max_val - min_val) if max_val > min_val else 0

        mean_thickness = np.mean(skeleton_thickness)
        std_thickness = np.std(skeleton_thickness)
        threshold = mean_thickness + std_multiplier * std_thickness
        obj_thresholded = np.where(raw_thickness_map[obj_mask] > threshold, raw_thickness_map[obj_mask], 0)

        temp_mask = np.zeros_like(mito_mask, dtype=np.float32)
        temp_mask[obj_mask] = obj_thresholded
        labeled_regions = label(temp_mask > 0, connectivity=2)
        props = regionprops(labeled_regions)

        for region in props:
            if region.area < min_region_size:
                continue

            centroid_xy = np.array([region.centroid[1], region.centroid[0]])
            dist, idx = kd_tree.query(centroid_xy.reshape(1, -1), k=1)
            closest_point = tree_points[idx[0]]
            distance_to_soma = np.nan
            branch_id = None
            branch_level = None
            branch_length = None
            for bid, polyline in interpolated_polylines.items():
                match_idx = np.where(np.all(np.isclose(polyline, closest_point, atol=1e-3), axis=1))[0]
                if len(match_idx) > 0:
                    distance_to_soma, branch_level = trace_to_soma(bid, match_idx[0], interpolated_polylines)
                    branch_id = bid
                    branch_length = cumulative_distances(dendrites[bid])[-1]
                    break

            region_coords = region.coords
            max_thickness = raw_thickness_map[region_coords[:, 0], region_coords[:, 1]].max()
            for coord in region_coords:
                thresholded_map[coord[0], coord[1]] = raw_thickness_map[coord[0], coord[1]]

            region_stats.append({
                "mito_ID": obj_id,
                "region_ID": region.label,
                "mito_area": np.sum(obj_mask),
                "region_area": region.area,
                "mean_local_width": mean_thickness,
                "max_local_width": max_thickness,
                "region_x": region.centroid[1],
                "region_y": region.centroid[0],
                "dist_soma": distance_to_soma,
                "branch_id": branch_id,
                "branch_level": branch_level,
                "branch_length": branch_length,
                "mito_length": mito_length
            })

    region_stats_df = pd.DataFrame(region_stats)
    return thickness_map, thresholded_map, binary_skeleton_map, region_stats_df






    

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from skimage.morphology import skeletonize, binary_erosion
from skimage.draw import disk
from skimage.measure import label, regionprops
import networkx as nx

def extract_longest_path_skeleton(skeleton):
    coords = np.column_stack(np.where(skeleton))
    G = nx.Graph()
    for y, x in coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < skeleton.shape[0] and 0 <= nx_ < skeleton.shape[1]:
                    if skeleton[ny, nx_]:
                        dist = np.sqrt(dy ** 2 + dx ** 2)
                        G.add_edge((y, x), (ny, nx_), weight=dist)
    if len(G.nodes) == 0:
        return np.zeros_like(skeleton, dtype=bool)
    endpoints = [n for n in G.nodes if G.degree[n] == 1]
    if len(endpoints) < 2:
        return skeleton
    max_path = []
    max_length = 0
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            try:
                path = nx.shortest_path(G, endpoints[i], endpoints[j], weight='weight')
                path_length = sum(np.linalg.norm(np.subtract(path[k + 1], path[k])) for k in range(len(path) - 1))
                if path_length > max_length:
                    max_length = path_length
                    max_path = path
            except nx.NetworkXNoPath:
                continue
    longest_mask = np.zeros_like(skeleton, dtype=bool)
    for y, x in max_path:
        longest_mask[y, x] = True
    return longest_mask

def calculate_skeleton_length(skeleton):
    coords = np.column_stack(np.where(skeleton))
    total_length = 0.0
    neighbors_offset = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [ 0, -1],          [ 0, 1],
        [ 1, -1], [ 1, 0], [ 1, 1]
    ])
    for coord in coords:
        neighbors = coord + neighbors_offset
        valid = (
            (neighbors[:, 0] >= 0) & (neighbors[:, 0] < skeleton.shape[0]) &
            (neighbors[:, 1] >= 0) & (neighbors[:, 1] < skeleton.shape[1])
        )
        valid_neighbors = neighbors[valid]
        valid_neighbors = valid_neighbors[skeleton[valid_neighbors[:, 0], valid_neighbors[:, 1]]]
        distances = np.sqrt(np.sum((valid_neighbors - coord) ** 2, axis=1))
        total_length += np.sum(distances) / 2
    return total_length

def remodeled_mito_area(
    mito_mask, root_nodes, dendrites, std_multiplier=1.0,
    skeleton_threshold=10, num_nearest=5, min_region_size=0, step=1
):
    def cumulative_distances(polyline):
        return np.cumsum([0] + [np.linalg.norm(polyline[i + 1] - polyline[i]) for i in range(len(polyline) - 1)])

    def interpolate_polyline(polyline):
        if len(polyline) < 2:
            return polyline
        distances = cumulative_distances(polyline)
        interpolator = interp1d(distances, polyline, axis=0, kind="linear", assume_sorted=True)
        dense_distances = np.arange(0, distances[-1], step)
        return interpolator(dense_distances)

    def build_kd_tree_and_map():
        interpolated = {bid: interpolate_polyline(p) for bid, p in dendrites.items()}
        points = np.vstack(list(interpolated.values()))
        tree = cKDTree(points)
        return tree, points, interpolated

    def trace_to_soma(branch_id, point_idx, interpolated):
        def collect_tree_nodes(node):
            return [node] + sum([collect_tree_nodes(child) for child in node.children], [])
        node_map = {n.branch_id: n for root in root_nodes for n in collect_tree_nodes(root)}
        polyline = interpolated[branch_id]
        distance = cumulative_distances(polyline)[point_idx]
        level = 0
        node = node_map.get(branch_id, None)
        while node and getattr(node, "parent", None):
            parent = node.parent
            parent_poly = interpolated[parent.branch_id]
            parent_dist = cumulative_distances(parent_poly)
            join_idx = np.argmin(np.linalg.norm(parent_poly - polyline[0], axis=1))
            distance += parent_dist[join_idx]
            polyline = parent_poly
            node = parent
            level += 1
        return distance, level

    kd_tree, tree_points, interpolated_polylines = build_kd_tree_and_map()

    labeled_mito = label(mito_mask, connectivity=2)
    area_map = np.zeros_like(mito_mask, dtype=np.float32)
    raw_area_map = np.zeros_like(mito_mask, dtype=np.float32)
    thresholded_map = np.zeros_like(mito_mask, dtype=np.float32)
    binary_skeleton_map = np.zeros_like(mito_mask, dtype=bool)
    region_stats = []

    for obj_id in range(1, labeled_mito.max() + 1):
        obj_mask = (labeled_mito == obj_id)
        skeleton = skeletonize(obj_mask)
        if np.sum(skeleton) < skeleton_threshold:
            continue

        binary_skeleton_map |= skeleton

        length_skeleton = extract_longest_path_skeleton(skeleton)
        mito_length = calculate_skeleton_length(length_skeleton)

        object_points = np.column_stack(np.where(obj_mask))
        skeleton_points = np.column_stack(np.where(skeleton))
        skeleton_tree = cKDTree(skeleton_points)
        _, indices = skeleton_tree.query(object_points)

        skeleton_to_pixels = {tuple(sp): [] for sp in skeleton_points}
        for pixel, sk_idx in zip(object_points, indices):
            skeleton_to_pixels[tuple(skeleton_points[sk_idx])].append(tuple(pixel))

        border_mask = obj_mask & ~binary_erosion(obj_mask)
        border_pixels = np.column_stack(np.where(border_mask))
        if len(border_pixels) > 0:
            border_tree = cKDTree(border_pixels)

        skeleton_areas = []
        for sk_point, mapped_pixels in skeleton_to_pixels.items():
            mapped_pixels = np.array(mapped_pixels)
            if len(border_pixels) > 0:
                distances, _ = border_tree.query(sk_point, k=min(num_nearest, len(border_pixels)))
                furthest = np.max(distances)
                rr, cc = disk((sk_point[0], sk_point[1]), furthest, shape=mito_mask.shape)
                local_area = np.sum(obj_mask[rr, cc])
                for pixel in mapped_pixels:
                    raw_area_map[pixel[0], pixel[1]] = local_area
                skeleton_areas.append(local_area)
            else:
                for pixel in mapped_pixels:
                    raw_area_map[pixel[0], pixel[1]] = 0
                skeleton_areas.append(0)

        obj_raw_values = raw_area_map[obj_mask]
        max_val = obj_raw_values.max()
        min_val = obj_raw_values.min()
        area_map[obj_mask] = (obj_raw_values - min_val) / (max_val - min_val) if max_val > min_val else 0

        mean_area = np.mean(skeleton_areas)
        std_area = np.std(skeleton_areas)
        threshold = mean_area + std_multiplier * std_area
        obj_thresholded = np.where(raw_area_map[obj_mask] > threshold, raw_area_map[obj_mask], 0)

        temp_mask = np.zeros_like(mito_mask, dtype=np.float32)
        temp_mask[obj_mask] = obj_thresholded
        labeled_regions = label(temp_mask > 0, connectivity=2)
        props = regionprops(labeled_regions)

        for region in props:
            if region.area < min_region_size:
                continue

            centroid_xy = np.array([region.centroid[1], region.centroid[0]])
            dist, idx = kd_tree.query(centroid_xy.reshape(1, -1), k=1)
            closest_point = tree_points[idx[0]]
            distance_to_soma = np.nan
            branch_id = None
            branch_level = None
            branch_length = None
            for bid, polyline in interpolated_polylines.items():
                match_idx = np.where(np.all(np.isclose(polyline, closest_point, atol=1e-3), axis=1))[0]
                if len(match_idx) > 0:
                    distance_to_soma, branch_level = trace_to_soma(bid, match_idx[0], interpolated_polylines)
                    branch_id = bid
                    branch_length = cumulative_distances(dendrites[bid])[-1]
                    break

            region_coords = region.coords
            max_area = raw_area_map[region_coords[:, 0], region_coords[:, 1]].max()
            for coord in region_coords:
                thresholded_map[coord[0], coord[1]] = raw_area_map[coord[0], coord[1]]

            region_stats.append({
                "mito_ID": obj_id,
                "region_ID": region.label,
                "mito_area": np.sum(obj_mask),
                "region_area": region.area,
                "mean_local_area": mean_area,
                "max_local_area": max_area,
                "region_x": region.centroid[1],
                "region_y": region.centroid[0],
                "dist_soma": distance_to_soma,
                "branch_id": branch_id,
                "branch_level": branch_level,
                "branch_length": branch_length,
                "mito_length": mito_length
            })

    region_stats_df = pd.DataFrame(region_stats)
    return area_map, thresholded_map, binary_skeleton_map, region_stats_df







import numpy as np
import pandas as pd
from skimage.draw import polygon

def mito_under_synapse(
    PSD95_puncta_props,
    PSD95_image,
    Mito_mask,
    dendrites,
    radius=5,
    rect_length=50,
    rect_width=10
):
    """
    Extract mitochondrial area under PSD95 puncta with rectangle geometry retained.

    Parameters:
        PSD95_puncta_props (list of dict): Output from detect_puncta, containing:
            - 'centroid', 'area', 'branch_id', 'branch_level', 'distance_to_soma_px'
        PSD95_image (np.ndarray): PSD95 intensity image.
        Mito_mask (np.ndarray): Binary or labeled mitochondrial mask.
        dendrites (dict): Dictionary {branch_id: [(x, y), ...]}.
        radius (int): Radius for intensity sampling (optional).
        rect_length (int): Length of oriented rectangle.
        rect_width (int): Width of oriented rectangle.

    Returns:
        pd.DataFrame: With columns:
            - psd95_ID
            - psd95_x
            - psd95_y
            - psd95_area
            - branch_id
            - branch_level
            - dist_soma
            - mito_area
            - Rect_coords
    """
    results = []

    for idx, prop in enumerate(PSD95_puncta_props, start=1):
        y, x = prop["centroid"]
        branch_id = prop["branch_id"]
        branch_level = prop["branch_level"]
        branch_length = prop["branch_length_px"]
        dist_soma = prop["distance_to_soma_px"]
        psd95_area = prop["area"]

        if branch_id not in dendrites:
            continue

        polyline = np.array(dendrites[branch_id])
        closest_point = None
        min_distance = float("inf")

        for start, end in zip(polyline[:-1], polyline[1:]):
            seg_vec = end - start
            seg_len = np.linalg.norm(seg_vec)
            if seg_len == 0:
                continue
            to_point_vec = np.array([x, y]) - start
            projection = np.dot(to_point_vec, seg_vec) / seg_len**2
            projection = np.clip(projection, 0, 1)
            proj_point = start + projection * seg_vec
            distance = np.linalg.norm(proj_point - [x, y])
            if distance < min_distance:
                min_distance = distance
                closest_point = proj_point
                vector = np.array([x, y]) - proj_point

        if closest_point is None or np.linalg.norm(vector) == 0:
            continue

        vector /= np.linalg.norm(vector)
        perp_vector = np.array([-vector[1], vector[0]])

        corners = [
            closest_point + vector * (rect_length / 2) + perp_vector * (rect_width / 2),
            closest_point + vector * (rect_length / 2) - perp_vector * (rect_width / 2),
            closest_point - vector * (rect_length / 2) - perp_vector * (rect_width / 2),
            closest_point - vector * (rect_length / 2) + perp_vector * (rect_width / 2),
        ]
        corners = np.array(corners)

        rr_rect, cc_rect = polygon(corners[:, 1], corners[:, 0], shape=Mito_mask.shape)
        rect_mask = np.zeros_like(Mito_mask, dtype=bool)
        rect_mask[rr_rect, cc_rect] = True
        mito_area = np.sum(Mito_mask[rect_mask] > 0)

        results.append({
            "psd95_ID": idx,
            "psd95_x": x,
            "psd95_y": y,
            "psd95_area": psd95_area,
            "branch_id": branch_id,
            "branch_level": branch_level,
            "branch_length": branch_length,
            "dist_soma": dist_soma,
            "mito_area": mito_area,
            "Rect_coords": corners.tolist()
        })

    return pd.DataFrame(results)
    
    
    
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import regionprops, label
import pandas as pd
from collections import defaultdict

def calculate_density_profiles(
    PSD95_mask, Mito_mask, dendrites, root_nodes,
    resolution=1.0, bin_size=100, step_size=10
):
    def interpolate_polyline(polyline, resolution):
        diffs = np.diff(polyline, axis=0)
        lengths = np.linalg.norm(diffs, axis=1)
        cumulative_lengths = np.concatenate([[0], np.cumsum(lengths)])
        total_length = cumulative_lengths[-1]
        if total_length == 0:
            return polyline
        interpolated_lengths = np.arange(0, total_length, resolution)
        interpolated_points = np.empty((len(interpolated_lengths), 2))
        for i, target_length in enumerate(interpolated_lengths):
            segment_idx = np.searchsorted(cumulative_lengths, target_length) - 1
            segment_idx = np.clip(segment_idx, 0, len(polyline) - 2)
            segment_start = polyline[segment_idx]
            segment_end = polyline[segment_idx + 1]
            segment_length = cumulative_lengths[segment_idx + 1] - cumulative_lengths[segment_idx]
            t = (target_length - cumulative_lengths[segment_idx]) / segment_length
            interpolated_points[i] = segment_start + t * (segment_end - segment_start)
        return interpolated_points

    def get_branch_level(branch_id, root_nodes):
        def collect_tree_nodes(node):
            return [node] + sum([collect_tree_nodes(child) for child in node.children], [])
        node_map = {n.branch_id: n for root in root_nodes for n in collect_tree_nodes(root)}
        level = 0
        node = node_map.get(branch_id, None)
        while node and getattr(node, "parent", None):
            level += 1
            node = node.parent
        return level

    def get_branch_length(polyline):
        if len(polyline) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(polyline, axis=0), axis=1)))

    # Handle 3D PSD95 input
    if PSD95_mask.ndim == 3:
        PSD95_mask = np.max(PSD95_mask, axis=0)

    # Prepare interpolated dendrites
    interpolated_dendrites = {
        branch_id: interpolate_polyline(np.array(points), resolution)
        for branch_id, points in dendrites.items() if len(points) > 1
    }

    # Region properties and mask coordinates
    labeled_psd95 = label(PSD95_mask.astype(np.uint8))
    labeled_mito = label(Mito_mask.astype(np.uint8))

    psd95_regions = regionprops(labeled_psd95)
    mito_objects = [np.column_stack(np.where(labeled_mito == i)) for i in range(1, labeled_mito.max() + 1)]

    psd95_centroids = [np.array(r.centroid)[-2:] for r in psd95_regions]
    psd95_areas = [r.area for r in psd95_regions]

    branch_mito_mapping = defaultdict(list)

    # Assign mito objects to nearest branch using flipped polylines
    for obj_coords in mito_objects:
        branch_distances = {}
        for branch_id, polyline in interpolated_dendrites.items():
            tree = cKDTree(polyline[:, ::-1])  # flip to (row, col)
            distances, _ = tree.query(obj_coords)
            branch_distances[branch_id] = np.sum(distances < resolution)
        assigned_branch = max(branch_distances, key=branch_distances.get)
        branch_mito_mapping[assigned_branch].append(obj_coords)

    # Now compute density profiles
    profiles = []

    for branch_id, polyline in interpolated_dendrites.items():
        if len(polyline) < 2:
            continue

        tree = cKDTree(polyline[:, ::-1])  # flip to (row, col)
        level = get_branch_level(branch_id, root_nodes)
        branch_length = get_branch_length(polyline)

        psd95_number_counts = np.zeros(len(polyline), dtype=int)
        psd95_area_values = np.zeros(len(polyline), dtype=float)
        mito_area_values = np.zeros(len(polyline), dtype=float)

        for centroid, area in zip(psd95_centroids, psd95_areas):
            _, idx = tree.query(centroid)
            psd95_number_counts[idx] += 1
            psd95_area_values[idx] += area

        if branch_id in branch_mito_mapping:
            for obj_coords in branch_mito_mapping[branch_id]:
                _, indices = tree.query(obj_coords)
                unique, counts = np.unique(indices, return_counts=True)
                mito_area_values[unique] += counts

        # Use point-based arclengths (matching object_density_profile)
        diffs = np.diff(polyline, axis=0)
        lengths = np.linalg.norm(diffs, axis=1)
        arclengths = np.concatenate([[0], np.cumsum(lengths)])
        centers = np.arange(0, arclengths[-1], step_size)

        for bin_id, center in enumerate(centers):
            start = center - bin_size / 2
            end = center + bin_size / 2
            in_window = (arclengths[:-1] >= start) & (arclengths[:-1] < end)
            profiles.append({
                "branch_id": branch_id,
                "branch_level": level,
                "branch_length": branch_length,
                "distance_to_soma": center,
                "bin_id": bin_id,
                "psd95_number_density": np.sum(psd95_number_counts[:-1][in_window]) / bin_size,
                "psd95_area_density": np.sum(psd95_area_values[:-1][in_window]) / bin_size,
                "mito_area_density": np.sum(mito_area_values[:-1][in_window]) / bin_size
            })

    return pd.DataFrame(profiles), interpolated_dendrites

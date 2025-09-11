import os
import pickle
import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
# ==============================================================
# === Visualization Helpers
# ==============================================================
def display_curvature_colored_meshes(mesh1, mesh2, mesh3, curvatures,
                                     min_val=None, max_val=None, smoothing_sigma=1.0):
    """
    Display three meshes: mesh1 + mesh2 in white, mesh3 color-coded by curvature.
    
    Curvature values are Gaussian-smoothed and clipped to percentile or user-provided limits.
    """
    if min_val is None:
        min_val = np.percentile(curvatures, 5)
    if max_val is None:
        max_val = np.percentile(curvatures, 95)

    smoothed = gaussian_filter(curvatures, sigma=smoothing_sigma)
    clipped = np.clip(smoothed, min_val, max_val)
    mesh3['Curvature'] = clipped

    plotter = pv.Plotter(notebook=True)
    plotter.background_color = 'white'
    plotter.add_mesh(mesh1, color='white', opacity=0.2)
    plotter.add_mesh(mesh2, color='white', opacity=0.2)
    plotter.add_mesh(mesh3, scalars='Curvature', cmap='YlOrRd',
                     opacity=1, nan_color="white", show_scalar_bar=False)
    plotter.show()


def display_cristae(mesh1, mesh2, mesh3, curvatures, threshold,
                   smooth_boundaries=False, smoothing_sigma=1.0):
    """
    Display three meshes, with mesh3 color-coded above/below a curvature threshold.
    """
    if len(curvatures) != mesh3.n_points:
        raise ValueError("Curvature array length must match number of mesh vertices.")
    
    smoothed = gaussian_filter(curvatures, sigma=smoothing_sigma) if smooth_boundaries else curvatures.copy()

    color_array = np.zeros((mesh3.n_points, 3)) 
    color_array[smoothed > threshold] = [0.9686, 0.7686, 0.2627]  # yellow
    color_array[smoothed <= threshold] = [0.9686, 0.7686, 0.2627] # same for now

    mesh3_copy = mesh3.copy()
    mesh3_copy.point_data["Color"] = color_array

    plotter = pv.Plotter(notebook=True)
    plotter.background_color = 'white'
    plotter.add_mesh(mesh1, color='black', opacity=0.2)
    plotter.add_mesh(mesh2, color='black', opacity=0.2)
    plotter.add_mesh(mesh3_copy, scalars="Color", rgb=True, opacity=1)
    plotter.add_light(pv.Light(position=(1,1,1), light_type='headlight', intensity=0.4))
    plotter.show()
    
    
    
    
def display_junctions(mesh1, mesh2, mesh3, points,
                        mesh3_color=(0.9686, 0.7686, 0.2627),
                        point_color=(0.294, 0.0, 0.510),
                        point_radius=0.01, point_opacity=1.0):
    plotter = pv.Plotter(notebook=True)
    plotter.background_color = 'white'
    plotter.add_mesh(mesh1, color='black', opacity=0.05)
    plotter.add_mesh(mesh2, color='black', opacity=0.05)
    plotter.add_mesh(mesh3, color=mesh3_color, opacity=1.0)

    if len(points):
        spheres = [pv.Sphere(radius=point_radius, center=p,
                             theta_resolution=12, phi_resolution=12) for p in points]
        merged = spheres[0].merge(spheres[1:]) if len(spheres) > 1 else spheres[0]
        plotter.add_mesh(merged, color=point_color, opacity=point_opacity)

    plotter.add_light(pv.Light(position=(1, 1, 1), light_type='headlight', intensity=0.4))
    plotter.show()
    
    
    
    
import os
import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

# ==============================================================
# === Helper Functions
# ==============================================================

def display_all(OMS, IBS, CMS, ER, ERMCS, Ribo, curvatures, threshold,
                smooth_boundaries=False, smoothing_sigma=1.0):
    """
    Display all relevant meshes together with CMS color-coded by curvature threshold.
    """
    if len(curvatures) != CMS.n_points:
        raise ValueError("Curvature array length must match the number of mesh vertices.")

    smoothed_curvatures = (gaussian_filter(curvatures, sigma=smoothing_sigma)
                           if smooth_boundaries else curvatures.copy())

    # Color map: currently both sides yellow (can be updated later)
    color_array = np.zeros((CMS.n_points, 3))
    color_array[smoothed_curvatures > threshold] = [0.9686, 0.7686, 0.2627]
    color_array[smoothed_curvatures <= threshold] = [0.9686, 0.7686, 0.2627]

    mesh3_copy = CMS.copy()
    mesh3_copy.point_data["Color"] = color_array

    plotter = pv.Plotter(notebook=True)
    plotter.background_color = 'white'
    plotter.add_mesh(OMS, color='black', opacity=0.2)
    plotter.add_mesh(IBS, color='black', opacity=0.2)
    plotter.add_mesh(ER, color=(0, 0.447, 0.698), opacity=0.2)
    plotter.add_mesh(ERMCS, color=(0, 0.447, 0.698), opacity=1)
    plotter.add_mesh(Ribo, color=(0.780, 0.082, 0.522), opacity=1)
    plotter.add_mesh(mesh3_copy, scalars="Color", rgb=True, opacity=1)
    plotter.add_light(pv.Light(position=(1, 1, 1),
                               light_type='headlight',
                               intensity=0.4))
    plotter.show()
    
    
    
    
    
    
# ==============================================================
# === Visualization
# ==============================================================

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def display_ribosome_clusters(OMS, IBS, CMS, ER, ERMCS, ribosome_clusters,
                                   curvatures, threshold,
                                   smooth_boundaries=False, smoothing_sigma=1.0):
    """
    Display all relevant meshes together with CMS color-coded by curvature threshold
    and ribosome clusters colored randomly.
    """
    if len(curvatures) != CMS.n_points:
        raise ValueError("Curvature array length must match the number of mesh vertices.")

    # Smooth curvature if requested
    smoothed_curvatures = (gaussian_filter(curvatures, sigma=smoothing_sigma)
                           if smooth_boundaries else curvatures.copy())

    # Color CMS (same logic as before — both sides yellow for now)
    color_array = np.zeros((CMS.n_points, 3))
    color_array[smoothed_curvatures > threshold] = [0.9686, 0.7686, 0.2627]
    color_array[smoothed_curvatures <= threshold] = [0.9686, 0.7686, 0.2627]

    mesh3_copy = CMS.copy()
    mesh3_copy.point_data["Color"] = color_array

    # Set up plotter
    plotter = pv.Plotter(notebook=True)
    plotter.background_color = 'white'

    # Fixed-color meshes (same as display_all)
    plotter.add_mesh(OMS, color='black', opacity=0.2)
    plotter.add_mesh(IBS, color='black', opacity=0.2)
    plotter.add_mesh(ER, color=(0, 0.447, 0.698), opacity=0.2)
    plotter.add_mesh(ERMCS, color=(0, 0.447, 0.698), opacity=1)
    plotter.add_mesh(mesh3_copy, scalars="Color", rgb=True, opacity=1)

    # Random colors for ribosome clusters
    cmap = plt.get_cmap('hsv')
    num_ribo = len(ribosome_clusters)
    colors = [cmap(i / num_ribo)[:3] for i in range(num_ribo)]

    for ribo, color in zip(ribosome_clusters, colors):
        plotter.add_mesh(ribo, color=color, opacity=1.0)

    # Lighting
    plotter.add_light(pv.Light(position=(1, 1, 1),
                               light_type='headlight',
                               intensity=0.4))
    plotter.show()
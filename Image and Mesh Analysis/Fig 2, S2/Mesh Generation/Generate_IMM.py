import os
import numpy as np
import tifffile as tiff
from scipy.ndimage import label
from skimage.morphology import remove_small_objects
from meshparty import trimesh_vtk
import trimesh
from skimage.measure import marching_cubes

def pyvista_to_trimesh(pv_mesh):
    """
    Converts a PyVista PolyData mesh to a Trimesh object.

    Parameters:
        pv_mesh (pv.PolyData): PyVista mesh.

    Returns:
        trimesh.Trimesh: Converted Trimesh object.
    """
    # Extract vertices and faces
    vertices = pv_mesh.points
    faces = pv_mesh.faces.reshape(-1, 4)[:, 1:4]  # Remove first column (number of points per face)

    # Create Trimesh object
    return trimesh.Trimesh(vertices=vertices, faces=faces)

import trimesh
import numpy as np
from scipy.spatial import cKDTree

def remove_close_faces(mesh1, mesh2, threshold=0.1):
    """
    Removes faces from mesh1 where any vertex is within a given distance from mesh2.

    Parameters:
        mesh1 (trimesh.Trimesh): The mesh to filter.
        mesh2 (trimesh.Trimesh): The reference mesh.
        threshold (float): Distance threshold for removal.

    Returns:
        trimesh.Trimesh: Filtered mesh1 with distant faces retained.
    """

    # Build a KDTree for fast nearest neighbor search
    tree = cKDTree(mesh2.vertices)

    # Find the closest distance from each vertex in mesh1 to mesh2
    distances, _ = tree.query(mesh1.vertices, k=1)  # k=1 finds the nearest neighbor

    # Get boolean mask of vertices that are too close to mesh2
    close_vertices_mask = distances < threshold

    # Find faces where ANY vertex is too close
    faces_to_remove = np.any(close_vertices_mask[mesh1.faces], axis=1)

    # Keep only faces that are NOT too close
    remaining_faces = mesh1.faces[~faces_to_remove]

    # Create a new mesh with remaining faces
    filtered_mesh = trimesh.Trimesh(vertices=mesh1.vertices, faces=remaining_faces, process=False)

    return filtered_mesh

import trimesh
import numpy as np
from scipy.spatial import cKDTree

def separate_close_faces(mesh1, mesh2, threshold=0.1):
    """
    Separates mesh1 into two meshes:
    1. A mesh where faces near mesh2 are removed.
    2. A mesh containing only the removed close faces.

    Parameters:
        mesh1 (trimesh.Trimesh): The mesh to filter.
        mesh2 (trimesh.Trimesh): The reference mesh.
        threshold (float): Distance threshold for removal.

    Returns:
        (trimesh.Trimesh, trimesh.Trimesh): 
        - The first mesh has distant faces retained.
        - The second mesh contains only the removed close faces.
    """

    # Build a KDTree for fast nearest-neighbor search
    tree = cKDTree(mesh2.vertices)

    # Find the closest distance from each vertex in mesh1 to mesh2
    distances, _ = tree.query(mesh1.vertices, k=1)  # k=1 finds the nearest neighbor

    # Get boolean mask of vertices that are too close to mesh2
    close_vertices_mask = distances < threshold

    # Find faces where ANY vertex is too close
    faces_to_remove = np.any(close_vertices_mask[mesh1.faces], axis=1)

    # Separate the faces
    remaining_faces = mesh1.faces[~faces_to_remove]  # Faces to keep
    close_faces = mesh1.faces[faces_to_remove]  # Faces that were removed

    # Create two new meshes
    filtered_mesh = trimesh.Trimesh(vertices=mesh1.vertices, faces=remaining_faces, process=False)
    close_faces_mesh = trimesh.Trimesh(vertices=mesh1.vertices, faces=close_faces, process=False)

    return filtered_mesh, close_faces_mesh


import pyvista as pv
base_dir = r"./Demo Images/control 4"
filename = "IMS_remeshed.stl"
file_path = os.path.join(base_dir, filename)
component = pv.read(file_path)
print(type(component))
IMS = pyvista_to_trimesh(component)

filename = "IBM_remeshed.stl"
file_path = os.path.join(base_dir, filename)
component = pv.read(file_path)
print(type(component))
IBM = pyvista_to_trimesh(component)

filename = "BB_remeshed.stl"
file_path = os.path.join(base_dir, filename)
component = pv.read(file_path)
print(type(component))
BB = pyvista_to_trimesh(component)

import trimesh
IMS = trimesh.boolean.intersection([IMS, BB], engine="manifold")
IBM = trimesh.boolean.intersection([IBM, BB], engine="manifold")

#Find zero area faces and remove
face_areas = IMS.area_faces
zero_area_faces = np.where(face_areas == 0)[0]
print("Indices of zero-area faces:", zero_area_faces)
print("Number of zero-area faces:", len(zero_area_faces))

# Remove zero-area faces
if len(zero_area_faces) > 0:
    IMS.update_faces(np.setdiff1d(np.arange(len(IMS.faces)), zero_area_faces))
    IMS.remove_unreferenced_vertices()  # Clean up unreferenced vertices

#Find zero area faces and remove
face_areas = IBM.area_faces
zero_area_faces = np.where(face_areas == 0)[0]
print("Indices of zero-area faces:", zero_area_faces)
print("Number of zero-area faces:", len(zero_area_faces))

# Remove zero-area faces
if len(zero_area_faces) > 0:
    IBM.update_faces(np.setdiff1d(np.arange(len(IBM.faces)), zero_area_faces))
    IBM.remove_unreferenced_vertices()  # Clean up unreferenced vertices

IMM = trimesh.boolean.difference([IBM, IMS], engine="manifold")
output = "IMM.stl"
output_path = os.path.join(base_dir, output)
IMM.export(output_path)

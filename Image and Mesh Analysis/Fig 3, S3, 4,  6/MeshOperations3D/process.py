import os
import pickle
import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter


# ==============================================================
# === Conversion Utilities
# ==============================================================
def pyvista_to_trimesh(pv_mesh):
    """
    Convert a PyVista PolyData mesh into a Trimesh object.
    
    Parameters
    ----------
    pv_mesh : pv.PolyData
        Input PyVista mesh.
    
    Returns
    -------
    trimesh.Trimesh
        Converted Trimesh object with vertices and faces.
    """
    vertices = pv_mesh.points
    faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]  # Drop leading size column
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def trimesh_to_pyvista(tri_mesh):
    """
    Convert a Trimesh object into a PyVista PolyData mesh.
    
    Parameters
    ----------
    tri_mesh : trimesh.Trimesh
        Input Trimesh object.
    
    Returns
    -------
    pv.PolyData
        Equivalent PyVista mesh.
    """
    faces = np.hstack((np.full((len(tri_mesh.faces), 1), 3), tri_mesh.faces)).astype(np.int64)
    return pv.PolyData(tri_mesh.vertices, faces)


# ==============================================================
# === Mesh Processing Helpers
# ==============================================================
def remove_close_faces(mesh1, mesh2, threshold=0.1):
    """
    Remove faces from mesh1 where any vertex lies within a threshold distance of mesh2.
    
    Parameters
    ----------
    mesh1 : trimesh.Trimesh
        Mesh to filter.
    mesh2 : trimesh.Trimesh
        Reference mesh.
    threshold : float
        Distance cutoff. Faces closer than this are removed.
    
    Returns
    -------
    trimesh.Trimesh
        Filtered mesh with close faces removed.
    """
    tree = cKDTree(mesh2.vertices)
    distances, _ = tree.query(mesh1.vertices, k=1)
    close_vertices_mask = distances < threshold
    faces_to_remove = np.any(close_vertices_mask[mesh1.faces], axis=1)
    remaining_faces = mesh1.faces[~faces_to_remove]
    return trimesh.Trimesh(vertices=mesh1.vertices, faces=remaining_faces, process=False)


def remove_close_faces_with_curvature_pv(mesh1, mesh2, threshold=0.1, curvature_field='Curvature'):
    """
    Remove faces from a PyVista mesh (mesh1) that are too close to mesh2,
    while keeping vertex curvature values intact.
    
    Parameters
    ----------
    mesh1 : pv.PolyData
        Mesh with curvature values in point_data.
    mesh2 : pv.PolyData
        Reference mesh.
    threshold : float
        Distance cutoff for removing faces.
    curvature_field : str
        Key in point_data storing curvature values.
    
    Returns
    -------
    tuple
        (filtered_mesh, new_curvature) where:
        - filtered_mesh : pv.PolyData
            Mesh with close faces removed and curvature preserved.
        - new_curvature : np.ndarray
            Curvature values corresponding to surviving vertices.
    """
    tree = cKDTree(mesh2.points)
    distances, _ = tree.query(mesh1.points, k=1)
    close_vertices_mask = distances < threshold

    faces = mesh1.faces.reshape(-1, 4)[:, 1:]
    keep_face_mask = ~np.any(close_vertices_mask[faces], axis=1)
    filtered_faces = faces[keep_face_mask]

    used_vertex_indices = np.unique(filtered_faces)
    old_to_new = {old: new for new, old in enumerate(used_vertex_indices)}
    remapped_faces = np.vectorize(old_to_new.get)(filtered_faces)

    new_points = mesh1.points[used_vertex_indices]
    new_curvature = mesh1.point_data[curvature_field][used_vertex_indices]

    n_faces = remapped_faces.shape[0]
    flat_faces = np.hstack([np.full((n_faces, 1), 3), remapped_faces]).astype(np.int32).flatten()
    filtered_mesh = pv.PolyData(new_points, flat_faces)
    filtered_mesh.point_data[curvature_field] = new_curvature
    return filtered_mesh, new_curvature


def load_pickle(file_path):
    """
    Safely load Python objects from a pickle file.
    
    Parameters
    ----------
    file_path : str
        Path to .pkl file.
    
    Returns
    -------
    list
        Contents as a list (ensures consistent return type).
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            return list(data) if isinstance(data, (list, tuple, set)) else [data]
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return []
        
        
def separate_high_curvature_mesh(mesh, curvature_values, threshold):
    """
    Split a mesh into two parts: one above curvature threshold, one below.
    
    Returns
    -------
    tuple : (high_curvature_mesh, low_curvature_mesh)
    """
    if len(curvature_values) != mesh.n_points:
        raise ValueError("Curvature array length must match number of mesh vertices.")

    high_mask = curvature_values > threshold
    faces = mesh.faces.reshape(-1, 4)
    face_indices = faces[:, 1:]
    face_high_mask = np.any(high_mask[face_indices], axis=1)

    high_faces = faces[face_high_mask].flatten()
    low_faces = faces[~face_high_mask].flatten()

    high_mesh = pv.PolyData(mesh.points, high_faces) if len(high_faces) > 0 else None
    low_mesh = pv.PolyData(mesh.points, low_faces) if len(low_faces) > 0 else None
    return high_mesh, low_mesh







import os
import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import networkx as nx
import itertools
from collections import defaultdict, deque
from meshparty import trimesh_vtk
import vtk
# ==============================================================
# === Helper Functions
# ==============================================================

def separate_close_faces(mesh1, mesh2, threshold=0.1):
    """
    Split mesh1 into two meshes:
    1. A mesh with faces further than 'threshold' from mesh2.
    2. A mesh containing only the close faces.
    """
    tree = cKDTree(mesh2.vertices)
    distances, _ = tree.query(mesh1.vertices, k=1)
    close_vertices_mask = distances < threshold

    faces_to_remove = np.any(close_vertices_mask[mesh1.faces], axis=1)
    remaining_faces = mesh1.faces[~faces_to_remove]
    close_faces = mesh1.faces[faces_to_remove]

    filtered_mesh = trimesh.Trimesh(vertices=mesh1.vertices,
                                    faces=remaining_faces,
                                    process=False)
    close_faces_mesh = trimesh.Trimesh(vertices=mesh1.vertices,
                                       faces=close_faces,
                                       process=False)
    return filtered_mesh, close_faces_mesh
    


# -------------------------------------------------------------------------
# Edge and centroid utilities
# -------------------------------------------------------------------------
def get_edges_from_mesh(mesh):
    edges = set()
    for face in mesh.faces:
        verts = mesh.vertices[face]
        for i in range(3):
            edge = tuple(sorted((tuple(verts[i]), tuple(verts[(i + 1) % 3]))))
            edges.add(edge)
    return edges

def group_edges_into_components(edges):
    v2e = defaultdict(set)
    for e in edges:
        for v in e:
            v2e[v].add(e)
    visited, comps = set(), []
    for e in edges:
        if e in visited:
            continue
        q, comp = deque([e]), set()
        while q:
            cur = q.popleft()
            if cur in visited: 
                continue
            visited.add(cur)
            comp.add(cur)
            q.extend(v2e[cur[0]] | v2e[cur[1]])
        comps.append(comp)
    return comps

def split_component_by_geometry(edges, min_path_length=5, distance_threshold=0.005):
    G = nx.Graph(); G.add_edges_from(edges)
    for v1, v2 in itertools.combinations(G.nodes, 2):
        if G.has_edge(v1, v2):
            continue
        try:
            path = nx.shortest_path(G, v1, v2)
        except nx.NetworkXNoPath:
            continue
        if len(path) - 1 >= min_path_length:
            if np.linalg.norm(np.array(v1) - np.array(v2)) < distance_threshold:
                path_edges = {tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)}
                path_edges.add(tuple(sorted((v1, v2))))
                return [path_edges, edges - path_edges]
    return [edges]

def shared_edge_junctions(mesh1, mesh2, color=(1, 0, 0), line_width=10, opacity=1.0,
                      min_component_size=15, min_path_length=10, distance_threshold=0.01):
    edges = get_edges_from_mesh(mesh1) & get_edges_from_mesh(mesh2)
    if not edges:
        print("No exactly matching edges found.")
        return None, [], None

    retained, valid = [], []
    for comp in group_edges_into_components(edges):
        if len(comp) < min_component_size:
            continue
        for sub in split_component_by_geometry(comp, min_path_length, distance_threshold):
            if len(sub) >= min_component_size:
                retained.extend(sub)
                valid.append(sub)

    if not retained:
        print(f"No components met min_component_size = {min_component_size}.")
        return None, [], None

    # Edge actor
    points, lines, pid_map, pid = vtk.vtkPoints(), vtk.vtkCellArray(), {}, 0
    for e in retained:
        line = vtk.vtkLine()
        for i, p in enumerate(e):
            if p not in pid_map:
                points.InsertNextPoint(p)
                pid_map[p] = pid
                pid += 1
            line.GetPointIds().SetId(i, pid_map[p])
        lines.InsertNextCell(line)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    edge_actor = vtk.vtkActor()
    edge_actor.SetMapper(mapper)
    edge_actor.GetProperty().SetColor(color)
    edge_actor.GetProperty().SetLineWidth(line_width)
    edge_actor.GetProperty().SetOpacity(opacity)

    # Centroids
    centroids = [np.mean(np.array(list({v for e in c for v in e})), axis=0) for c in valid]
    centroid_actor = (trimesh_vtk.point_cloud_actor(np.array(centroids), color=(1, 0, 0), size=0.005)
                      if centroids else None)
    if centroid_actor:
        centroid_actor.GetProperty().SetOpacity(1.0)

    return edge_actor, centroids, centroid_actor

def filter_junctions(CMS, centroids, min_faces=500,
                                            point_color=(0.294, 0.0, 0.510),
                                            point_size=0.01, opacity=1.0):
    if not len(centroids) or not len(CMS.faces):
        return np.empty((0, 3)), None

    comps = CMS.split(only_watertight=False)
    trees = [(c, cKDTree(c.vertices)) for c in comps]

    refined = []
    for pt in centroids:
        comp = min(trees, key=lambda ct: ct[1].query(pt)[0])[0]
        if len(comp.faces) >= min_faces:
            refined.append(pt)

    refined = np.array(refined)
    actor = None
    if len(refined):
        actor = trimesh_vtk.point_cloud_actor(refined, color=point_color, size=point_size)
        actor.GetProperty().SetOpacity(opacity)
    return refined, actor
    











import numpy as np
import pandas as pd
import pyvista as pv
import trimesh
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


# ==============================================================
# === Ribosome Splitting + Merging
# ==============================================================

def separate_ribosomes_with_merge(ribo_mesh, distance_threshold=0.02):
    """
    Separates a ribosome mesh into disconnected components and merges components 
    that are within a specified distance of each other.

    Parameters
    ----------
    ribo_mesh : trimesh.Trimesh
        The combined ribosome mesh.
    distance_threshold : float
        Distance below which components are merged.

    Returns
    -------
    list of pv.PolyData
        List of ribosome meshes (PyVista PolyData) after merging.
    """
    ribosome_list = list(ribo_mesh.split(only_watertight=False))
    merged_flags = [False] * len(ribosome_list)
    final_components = []

    for i, ribo_i in enumerate(ribosome_list):
        if merged_flags[i]:
            continue

        combined_mesh = ribo_i.copy()
        merged_flags[i] = True
        tree_i = cKDTree(ribo_i.vertices)

        for j in range(i + 1, len(ribosome_list)):
            if merged_flags[j]:
                continue
            ribo_j = ribosome_list[j]
            dist_ij, _ = tree_i.query(ribo_j.vertices, k=1)
            if np.min(dist_ij) < distance_threshold:
                combined_mesh = trimesh.util.concatenate([combined_mesh, ribo_j])
                merged_flags[j] = True
                tree_i = cKDTree(combined_mesh.vertices)

        # Convert to PyVista
        faces = np.hstack((np.full((len(combined_mesh.faces), 1), 3),
                           combined_mesh.faces)).astype(np.int64)
        pv_mesh = pv.PolyData(combined_mesh.vertices, faces)
        final_components.append(pv_mesh)

    print(f"Final ribosome components after merging: {len(final_components)}")
    return final_components






# ==============================================================
# === Ribosome vs OMS Analysis
# ==============================================================

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree, ConvexHull
import pyvista as pv

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
import pyvista as pv
import trimesh

def analyze_ribosomes_vs_OMS(ribo_meshes, OMS):
    """
    Measures distances and properties for each ribosome mesh relative to OMS.
    Accepts both Trimesh and PyVista meshes.
    """
    # OMS points
    OMS_points = OMS.points if isinstance(OMS, pv.PolyData) else OMS.vertices
    oms_tree = cKDTree(OMS_points)
    oms_area = OMS.area

    data = []
    for idx, ribo in enumerate(ribo_meshes, start=1):
        # Ribosome points
        ribo_points = ribo.points if isinstance(ribo, pv.PolyData) else ribo.vertices
        if ribo_points.shape[0] == 0:
            continue

        # Surface distance
        dists, _ = oms_tree.query(ribo_points, k=1)
        min_distance = dists.min()

        # Centroid distance
        centroid = ribo.center if isinstance(ribo, pv.PolyData) else ribo.centroid
        centroid_distance, _ = oms_tree.query(centroid, k=1)

        # Volume
        if isinstance(ribo, pv.PolyData):
            try:
                ribo_volume = ribo.volume
            except Exception:
                try:
                    hull = ConvexHull(ribo_points)
                    ribo_volume = hull.volume
                except Exception:
                    ribo_volume = np.nan
        else:  # trimesh
            try:
                ribo_volume = ribo.volume
            except Exception:
                try:
                    hull = ConvexHull(ribo_points)
                    ribo_volume = hull.volume
                except Exception:
                    ribo_volume = np.nan

        data.append({
            'Ribosome_ID': idx,
            'Min_Surface_Distance': min_distance * 1e3,   # nm
            'Centroid_Distance': centroid_distance * 1e3, # nm
            'Ribosome_Volume': ribo_volume * 1e9,         # nm³
            'OMS_Surface_Area': oms_area * 1e6            # nm²
        })

    df = pd.DataFrame(data)
    print(f"Analyzed {len(df)} clusters.")
    return df
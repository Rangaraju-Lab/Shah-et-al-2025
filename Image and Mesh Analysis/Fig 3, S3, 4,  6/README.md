# 3D Mesh Analysis Pipeline

The 3D Mesh Analysis Pipeline used in this study is a customized implementation integrating functions from **Trimesh**, **PyVista**, and **MeshParty** to visualize and quantify **structural properties** from 3D meshes of mitochondrial membranes, endoplasmic reticulum (ER), and ribosomes.

The primary meshes generated from  
```
Image and Mesh Analysis/Fig 2, S2/Mesh Generation
```
were imported into **Blender** and remeshed using voxelization. This remeshing creates a smooth manifold mesh which is essential for proper curvature measurement.
The **first principal curvature** of cristae was calculated using the **GAMer2** add-on in Blender by *Lee et al., 2020*, similar to *Mendelsohn et al., 2021*.

Most of the custom functions are stored within the **`MeshOperations3D`** module.  
The Jupyter notebook **`Mesh Visualization and Quantification.ipynb`** demonstrates the end-to-end process for visualization and quantification using Blender-processed meshes.

---

## Quick Demo (3D Mesh Analysis)

For a quick review of the 3D mesh analysis workflow:

1. Make sure the **GitHub code repository** and **Edmond data repository** are merged into a common folder.  
2. Open the Jupyter notebook  
   ```
   Mesh Visualization and Quantification.ipynb
   ```
3. Run all code blocks in the notebook.

This will display interactive meshes of **mitochondrial membranes**, **cristae junctions**, **ER**, **ERMCS**, and **ribosomes** within the notebook and print structural quantifications.

- **Mitochondrial volume**
- **Cristae surface area**
- **High-curvature surface area**
- **Cristae junctions and density**
- **ER–mitochondria contact site (ERMCS) surface area**
- **ER surface area**
- **Ribosome cluster dataframes**
- **Ribosome distances from OMM**

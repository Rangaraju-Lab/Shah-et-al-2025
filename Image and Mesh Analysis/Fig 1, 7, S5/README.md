# Confocal Image Analysis Pipeline

The Confocal Image Analysis Pipeline used in this study is a customized Python implementation designed to segment and quantify **structural properties of mitochondria and spines** along manually traced dendrites.

The manual dendritic traces were created in **ImageJ/Fiji** using the *Segmented Line Tool*, smoothed, interpolated, and saved as a `.csv` file containing coordinates. These files are used to replicate dendritic traces within a Python environment for localized image segmentation and quantification.

---

## Quick Demo (Confocal Image Analysis)

For a quick review of the confocal image analysis workflow:

1. Make sure the **GitHub code repository** and the **Edmond data repository** are merged into a common folder.  
2. Open the Jupyter notebook  
   ```
   Confocal Image Analysis.ipynb
   ```
3. Run the code block in the notebook.

This will save all **output segmentations** and **measurement files** in the same folder as the notebook.

---

## Workflow Overview

At its core, the workflow utilizes the following from *scikit-image*:
- **Otsu thresholding** for adaptive binarization of local image regions.
- **`blob_log` (Laplacian of Gaussian)** for puncta detection within the local neighborhood of the dendritic trace.

These algorithms are applied specifically within the region surrounding the dendritic path, allowing selective segmentation and quantification of structures.


---

## Output Files

- **Visualization Images:** Mitochondrial and PSD95 masks, mitochondrial skeleton, width and high-width maps, and overlays of mitochondria under spines.  
- **Quantification CSVs** Mitochondrial length, spine area, mitochondrial area within 1 µm of spines, and high-width mitochondrial region metrics.  

All outputs are automatically saved to the same directory as the notebook.

---

## Customization

The workflow within `Confocal Image Analysis.ipynb` is generalizable to segment structures along dendritic traces by adjusting parameters utilized by the functions from `Confocal2D` module.

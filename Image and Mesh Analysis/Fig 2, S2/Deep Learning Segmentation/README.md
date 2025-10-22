# Deep Learning Segmentation of Mitochondrial Intermembrane Space

The deep learning pipeline used in this study is a customized implementation that integrates concepts from **Suga et al., 2023** and the core network architecture from **Lee et al., 2017**.  
These scripts were run on a high-performance cluster computer with CUDA-enabled **NVIDIA A100 40 GB GPU** (FAU HPC resources) for model training, and on a CUDA-enabled **Windows 11 PC with NVIDIA GeForce RTX 3060 12 GB** for model application.

A complete set of codes to either train a fresh model or apply a model trained by us is provided.  
Our trained model segments the **intermembrane space** of mitochondria within the EM tomograms acquired for this study.

---

## Quick Demo (Model Application)

For a quick review of the model on a demo image:

1. Make sure the **GitHub code repository** and **Edmond data repository** are merged into a common folder.  
2. Open a terminal within  
   ```
   Image and Mesh Analysis/Fig 2, S2/Deep Learning Segmentation
   ```
3. Run  
   ```
   python .\Inference.py
   ```
4. Compare the results to files in **Our Output** to cross-reference.

To test cristae segmentation on other tomograms, modify the file paths inside `Inference.py`:

- Line 235  
  ```
  input_tomogram = "Control4.tif"
  ```
  should be changed to your desired file.

The output file is a prediction image with pixel intensities between **0–1**.  
This image was thresholded in **Fiji** to make all pixels above **0.2** foreground.  
The resulting binary segmentation was then **manually proofread**.

---

## Training a New Model

To train a fresh model:

1. Open a terminal within  
   ```
   Image and Mesh Analysis/Fig 2, S2/Deep Learning Segmentation
   ```
2. Run  
   ```
   python .\CristaeGPU.py
   ```

This initializes a process to first create training patches for all three axes and then commence training.

---

## Output Files

- **Training outputs** are stored in  
  ```
  Image and Mesh Analysis/Fig 2, S2/Deep Learning Segmentation/visual_outputs
  ```
- **Model checkpoints** are saved automatically whenever the **F1 score improves** or **validation loss decreases**.
- **Training metrics** (Training Loss, Validation Loss, IoU, Accuracy, Precision, Recall, F1) are stored in  
  ```
  training_metrics.csv
  ```

---

## Custom Training on New Data

The deep learning workflow is **generalizable** to structures other than mitochondrial intermembrane space.  
Instructions to create training data on your own images are provided within  
```
./Generate Training Data
```

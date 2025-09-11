if __name__ == '__main__':
    import os
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, random_split, ConcatDataset

    from Config import opt  # Import configuration options
    from Patches import Extract  
    from PaddedPatches import PaddedExtract
    from Augmentation import AugmentedDataset  # Augmentation wrapper


    # Create original extractors
    extractor1 = Extract(opt.data_dir, opt.label_dir, subpatch_size=(32, 128, 128), stride_ratio=(0.5, 0.5, 0.5))
    extractor2 = Extract(opt.data_dir, opt.label_dir, subpatch_size=(48, 32, 128), stride_ratio=(0.5, 1, 0.5))
    extractor3 = Extract(opt.data_dir, opt.label_dir, subpatch_size=(48, 128, 32), stride_ratio=(0.5, 0.5, 1))

    # Create PaddedExtract objects for each extractor
    padded_extractor1 = PaddedExtract(extractor1, target_shape=(32, 128, 128))
    padded_extractor2 = PaddedExtract(extractor2, target_shape=(32, 48, 128))
    padded_extractor3 = PaddedExtract(extractor3, target_shape=(32, 48, 128))


    # Example: Retrieve and print the first patch from each padded extractor
    data1, label1, mask1 = padded_extractor1[0]
    data2, label2, mask2 = padded_extractor2[0]
    data3, label3, mask3 = padded_extractor3[0]

    # Print shapes to confirm
    print(f"Patches in XY Plane: {len(padded_extractor1)}, Patch Shape: {data1.shape}")
    print(f"Patches in XZ Plane: {len(padded_extractor2)}, Patch Shape: {data2.shape}")
    print(f"Patches in YZ Plane: {len(padded_extractor3)}, Patch Shape: {data3.shape}")
    # Display the first patch from each padded extractor
    #padded_extractor1.display_patch(0)
    #padded_extractor2.display_patch(0)
    #padded_extractor3.display_patch(0)

    # Define the split ratio
    train_ratio = 0.8

    # Function to split a dataset into train and validation subsets
    def split_dataset(dataset, train_ratio):
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        return random_split(dataset, [train_size, val_size])

    # Split each padded extractor into training and validation sets
    train_extractor1, val_extractor1 = split_dataset(padded_extractor1, train_ratio)
    train_extractor2, val_extractor2 = split_dataset(padded_extractor2, train_ratio)
    train_extractor3, val_extractor3 = split_dataset(padded_extractor3, train_ratio)

    # Wrap with augmentation for training, no augmentation for validation
    train_dataset1 = AugmentedDataset(train_extractor1, augment=True)
    val_dataset1 = AugmentedDataset(val_extractor1, augment=False)

    train_dataset2 = AugmentedDataset(train_extractor2, augment=True)
    val_dataset2 = AugmentedDataset(val_extractor2, augment=False)

    train_dataset3 = AugmentedDataset(train_extractor3, augment=True)
    val_dataset3 = AugmentedDataset(val_extractor3, augment=False)

    # Create DataLoader objects for each train/validation dataset
    train_loader1 = DataLoader(train_dataset1, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader1 = DataLoader(val_dataset1, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    train_loader2 = DataLoader(train_dataset2, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader2 = DataLoader(val_dataset2, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    train_loader3 = DataLoader(train_dataset3, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader3 = DataLoader(val_dataset3, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    #Confirm the setup by printing the sizes of each set
    print(f"Total patches 1: {len(padded_extractor1)}")
    print(f"Training patches 1: {len(train_dataset1)}")
    print(f"Validation patches 1: {len(val_dataset1)}")

    #Confirm the setup by printing the sizes of each set
    print(f"Total patches 2: {len(padded_extractor2)}")
    print(f"Training patches 2: {len(train_dataset2)}")
    print(f"Validation patches 2: {len(val_dataset2)}")
    
    #Confirm the setup by printing the sizes of each set
    print(f"Total patches 3: {len(padded_extractor3)}")
    print(f"Training patches 3: {len(train_dataset3)}")
    print(f"Validation patches 3: {len(val_dataset3)}")
    
    
    import tifffile as tiff
    from Visualize import save_slices
    # Load the visual patch from a TIFF file
    visual_patch_path = "./visual_patch.tif"  # Replace with your actual file path
    visual_patch = tiff.imread(visual_patch_path)

    # Add a channel dimension if missing (to match shape: (1, Z, Y, X))
    if visual_patch.ndim == 3:
        visual_patch = np.expand_dims(visual_patch, axis=0)

    # Apply Z-score normalization
    mean = np.mean(visual_patch)
    std = np.std(visual_patch) + 1e-8  # Add epsilon to avoid division by zero
    visual_patch = (visual_patch - mean) / std

    # Print sanity checks
    print(f"Loaded visual patch shape: {visual_patch.shape}")  # Should be (1, Z, Y, X)
    print(f"Visual Patch - min: {visual_patch.min()}, max: {visual_patch.max()}")

    from ResidualUNet import ResidualUNet
    from Loss import RebalancedMaskedBCELoss
    import torch.optim as optim

    # Initialize the model
    model = ResidualUNet(
        in_channels=1, out_channels=1
    )

    # Move the model to the appropriate device (GPU or CPU)
    model = model.to(opt.device)
    
    # Initialize the loss function with optional max_weight
    criterion = RebalancedMaskedBCELoss(size_average=True, max_weight=10.0)

    # Initialize the optimizer with Adam and add weight decay
    optimizer = optim.Adam(
        model.parameters(), lr=opt.base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5
    )

    # ReduceLROnPlateau scheduler with an adjusted minimum learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6
    )


    from Run import train_and_validate  # Import the training function

    # Start the training and validation process with separate loaders for each patch type
    train_and_validate(
        model=model,
        train_loaders=[train_loader1, train_loader2, train_loader3],
        val_loaders=[val_loader1, val_loader2, val_loader3],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        visual_patch=visual_patch,
        opt=opt
    )
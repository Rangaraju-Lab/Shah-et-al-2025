import os
import numpy as np
import torch
import csv
from tqdm import tqdm
from Visualize import save_slices
from torch.amp import GradScaler, autocast
import torch.nn.utils as utils
from torchmetrics.classification import BinaryJaccardIndex, BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score

def train_and_validate(model, train_loaders, val_loaders, criterion, optimizer, scheduler, visual_patch, opt, override_lr=None):
    """Train and validate the model with multiple loaders for different patch sizes, and log metrics to a CSV file."""
    model.to(opt.device)
    best_val_loss = float('inf')
    best_f1 = 0.0
    start_epoch = 0
    clip_value = 1.0

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler('cuda')
    
    # Initialize metrics on the specified device
    iou_metric = BinaryJaccardIndex().to(opt.device)
    accuracy_metric = BinaryAccuracy().to(opt.device)
    precision_metric = BinaryPrecision().to(opt.device)
    recall_metric = BinaryRecall().to(opt.device)
    f1_metric = BinaryF1Score().to(opt.device)

    # Ensure visual patch has batch dimension (1, C, Z, Y, X)
    if visual_patch.ndim == 4:  
        visual_patch = np.expand_dims(visual_patch, axis=0)

    visual_patch = torch.tensor(visual_patch, dtype=torch.float32).to(opt.device)

    # Initialize CSV for metrics logging
    csv_file_path = os.path.join(opt.output_dir, 'training_metrics.csv')
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Epoch', 'Patch Type', 'Train Loss', 'Val Loss', 'IoU', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

    checkpoint_path = os.path.join(opt.output_dir, "best_model.pth")
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=opt.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        best_f1 = checkpoint.get('best_f1', 0.0)
        print(f"Resuming from epoch {start_epoch + 1} with best validation loss {best_val_loss:.4f} and best F1 Score {best_f1:.4f}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    for epoch in range(start_epoch, opt.epochs):
        print(f"Epoch {epoch + 1}/{opt.epochs}")
        
        if override_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = override_lr
            print(f"Overriding learning rate to: {override_lr:.6f}")
        else:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Current Learning Rate: {current_lr:.6f}")

        # Iterate through each pair of train and validation loaders (one for each patch type)
        for i, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders), start=1):
            patch_type = f"Patch Type {i}"
            
            # Training Phase
            model.train()
            train_loss = 0.0

            for data, label, mask in tqdm(train_loader, desc=f"Training {patch_type}", ascii=True, dynamic_ncols=False):
                data, label, mask = (
                    data.to(opt.device), 
                    label.to(opt.device), 
                    mask.to(opt.device)
                )

                optimizer.zero_grad()
                with autocast('cuda'):
                    output = model(data)
                    loss = criterion(output, label, mask)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                utils.clip_grad_norm_(model.parameters(), clip_value)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            # Validation Phase
            model.eval()
            val_loss = 0.0

            # Reset metrics at the start of validation epoch
            iou_metric.reset()
            accuracy_metric.reset()
            precision_metric.reset()
            recall_metric.reset()
            f1_metric.reset()

            with torch.no_grad():
                for data, label, mask in tqdm(val_loader, desc=f"Validation {patch_type}", ascii=True, dynamic_ncols=False):
                    data, label, mask = (
                        data.to(opt.device), 
                        label.to(opt.device), 
                        mask.to(opt.device)
                    )

                    with autocast('cuda'):
                        output = model(data)
                        loss = criterion(output, label, mask)
                    val_loss += loss.item()

                    preds = (output > 0.5).float()
                    iou_metric.update(preds, label)
                    accuracy_metric.update(preds, label)
                    precision_metric.update(preds, label)
                    recall_metric.update(preds, label)
                    f1_metric.update(preds, label)

            # Calculate and print epoch metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            iou = iou_metric.compute().item()
            accuracy = accuracy_metric.compute().item()
            precision = precision_metric.compute().item()
            recall = recall_metric.compute().item()
            f1 = f1_metric.compute().item()

            print(f"Epoch {epoch + 1} - {patch_type} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"IoU: {iou:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

            # Log metrics to CSV
            with open(csv_file_path, mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([epoch + 1, patch_type, train_loss, val_loss, iou, accuracy, precision, recall, f1])

            # Update scheduler based on validation loss
            scheduler.step(val_loss)

            # Infer on the visual patch after each loader validation
            with torch.no_grad():
                with autocast('cuda'):
                    prediction = model(visual_patch)

            # Save the model if validation loss improves or if F1 Score improves
            if val_loss < best_val_loss or f1 > best_f1:
                best_val_loss = min(val_loss, best_val_loss)
                best_f1 = max(f1, best_f1)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_f1': best_f1
                }
                model_path = os.path.join(opt.output_dir, "best_model.pth")
                torch.save(checkpoint, model_path)
                print(f"Saved best model to {model_path}")

                # Save visualization
                save_slices(
                    em_image=visual_patch.cpu().numpy(), 
                    prediction=prediction.cpu().numpy(), 
                    epoch=epoch + 1, 
                    train_loss=train_loss, 
                    val_loss=val_loss, 
                    output_dir=opt.output_dir
                )

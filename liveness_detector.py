import os
import shutil
from tabnanny import verbose
import cv2
import pandas as pd
import random
import zipfile
import numpy as np
from skimage.transform import resize
from traitlets import Bunch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
import torch
import torch.nn as nn
from torchvision import models
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score


# Custom Dataset to hold the processed results output compatible with DataLoader
class LivenessDatasetWrapper(Dataset):
    def __init__(self, processed_data):
        self.source = processed_data
        self.is_dataset = False

    def __len__(self):
        if self.is_dataset:
            return len(self.source)
        return len(self.source.images)

    def __getitem__(self, idx):
        if self.is_dataset:
            return self.source[idx]
        img = self.source.images[idx]
        lbl = self.source.target[idx]
        
        # Ensure Tensor format (C, H, W)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        
        if not isinstance(lbl, torch.Tensor):
            lbl = torch.tensor(lbl, dtype=torch.long)
            
        return img, lbl
    
    
    
class LivenessMobileNet(nn.Module):
    def __init__(self, pretrained=True):
        super(LivenessMobileNet, self).__init__()
        # Load Pretrained MobileNetV2
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)
        
        # Replace the classifier head
        # Original input features of classifier[1] is 1280
        # output 1 value (Logit) for Binary Classification
        self.backbone.classifier[1] = nn.Linear(in_features=1280, out_features=1)
        
    def forward(self, x):
        return self.backbone(x)
    
    
    



def download_liveness_dataset(data_dir, train_split=0.7, val_split=0.15, test_split=0.15):
    """
    Loads the Liveness Dataset (Real vs Fake) keeping original resolution.
    Merges 'galleries' and 'probes' folders and splits them globally.
    store only the file paths (strings) in dataset lists 
    Ensures reproducibility with fixed random seed.
    Structure Expected:
    data_dir/
       galleries/
          fake/subject_XX/...
          real/subject_XX/...
       probes/
          fake/subject_XX/...
          real/subject_XX/...
    Args:
        data_dir (str): Path to root folder.
        train_split (float): Ratio for training.
        val_split (float): Ratio for validation.
        test_split (float): Ratio for testing.

    Returns:
        tuple: (train_data, val_data, test_data, target_names)
               Each *_data is a Bunch with .images (numpy array of objects) and .target
    """

    classes = {'fake': 0, 'real': 1}
    target_names = np.array(['Fake', 'Real'])
    all_paths = [] # Store strings, not arrays
    all_labels = []
    all_subjects = []               # Store the subject ID (e.g., "subject_01")
    sources = ['galleries', 'probes']
    
    for source in sources:
        source_path = os.path.join(data_dir, source)
        if not os.path.exists(source_path): continue
        
        for class_name, label_id in classes.items():
            class_path = os.path.join(source_path, class_name)
            if not os.path.exists(class_path): 
                print(f"Warning: Class path '{class_path}' does not exist. Skipping.")
                continue
            
            for subject_folder in os.listdir(class_path):
                subject_path = os.path.join(class_path, subject_folder)
                if os.path.isdir(subject_path):
                    for file in os.listdir(subject_path):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            img_path = os.path.join(subject_path, file)
                            # Store only the paths
                            all_paths.append(img_path)
                            all_labels.append(label_id)
                            all_subjects.append(subject_folder)            
                            
    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)
    all_subjects = np.array(all_subjects)
    print(f"Found {len(all_paths)} images across {len(np.unique(all_subjects))} subjects.")
    
    # Perform Subject-Disjoint Split using GroupShuffleSplit
    # This splits the GROUPS (Subjects), not just the images.
    # First Split: Train vs (Val + Test)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_split, random_state=42)
    train_idx, temp_idx = next(gss1.split(all_paths, all_labels, groups=all_subjects))
    
    # Second Split: Val vs Test
    # Adjust ratio for the remaining data
    relative_test_size = test_split / (val_split + test_split)
    # Get the data for the second split
    temp_paths = all_paths[temp_idx]
    temp_labels = all_labels[temp_idx]
    temp_subjects = all_subjects[temp_idx]
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test_size, random_state=42)
    val_sub_idx, test_sub_idx = next(gss2.split(temp_paths, temp_labels, groups=temp_subjects))
    
    # Map back to original indices
    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

    # 4. Create Bunches
    def create_bunch(idxs, split_name):
        paths = all_paths[idxs]
        lbls = all_labels[idxs]
        # Verify no subject leakage
        unique_subs = np.unique(all_subjects[idxs])
        return Bunch(images=paths, target=lbls, target_names=target_names, subjects=unique_subs)

    train_data = create_bunch(train_idx, "Training")
    val_data   = create_bunch(val_idx,   "Validation")
    test_data  = create_bunch(test_idx,  "Test")
    
    # Double Check Leakage
    train_subs = set(train_data.subjects)
    val_subs = set(val_data.subjects)
    test_subs = set(test_data.subjects)
    
    if train_subs.intersection(val_subs) or train_subs.intersection(test_subs):
        print("CRITICAL WARNING: DATA LEAKAGE DETECTED (Subjects overlap)!")
    else:
        print("Success: No subject overlap between splits.")

    return train_data, val_data, test_data, target_names






def evaluate_liveness(loader, model, device, threshold=0.5, live_demo=False):
    """
    Evaluates the liveness model on the provided DataLoader and computes metrics.
    Args:
        loader (DataLoader): DataLoader for the liveness dataset.
        model: Trained liveness model.
        device: 'cuda' or 'cpu'.
        threshold (float): Score above which a face is considered live.
        live_demo (bool): If True, skips visualization (for live demos).
    Returns:
        tuple: (all_probs, all_labels, flr, ffr)
    """
    model.eval()
    model.to(device)
    all_probs = []
    all_labels = []

    # 1. Inference Loop
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().view(-1).numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    predictions = (all_probs >= threshold).astype(int)

    # 2. Compute Metrics
    acc = accuracy_score(all_labels, predictions)
    fake_indices = (all_labels == 0)
    real_indices = (all_labels == 1)
    
    flr = np.mean(predictions[fake_indices] == 1) if np.sum(fake_indices) > 0 else 0.0
    ffr = np.mean(predictions[real_indices] == 0) if np.sum(real_indices) > 0 else 0.0
    acer = (flr + ffr) / 2

    if not live_demo:
        # 3. Visualization Logic
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # --- Plot A: Score Distribution ---
        ax_hist = axes[0]
        if np.any(real_indices):
            ax_hist.hist(all_probs[real_indices], bins=30, alpha=0.5, color='green', label='Live', density=True)
        if np.any(fake_indices):
            ax_hist.hist(all_probs[fake_indices], bins=30, alpha=0.5, color='red', label='Fake', density=True)
            
        ax_hist.axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Tf={threshold:.3f}')
        ax_hist.set_title("Score Distribution", fontweight='bold')
        ax_hist.set_xlabel("Liveness Probability")
        ax_hist.legend()
        ax_hist.grid(alpha=0.3)

        # --- Plot B: ROC Curve ---
        if len(np.unique(all_labels)) > 1:
            fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs)
            roc_auc = auc(fpr, tpr)
            ax_roc = axes[1]
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            
            # Mark current threshold on ROC (Nearest neighbor search)
            if len(roc_thresholds) > 0:
                idx = np.argmin(np.abs(roc_thresholds - threshold))
                ax_roc.scatter(fpr[idx], tpr[idx], color='black', zorder=5, label='Current Tf')
                
            ax_roc.set_title("ROC Curve", fontweight='bold')
            ax_roc.set_xlabel('FAR (False Accept Rate)')
            ax_roc.set_ylabel('TPR (True Accept Rate)')
            ax_roc.legend()
            ax_roc.grid(alpha=0.3)

        # --- Plot C: FAR & FRR vs Threshold (CORRECTED) ---
        ax_err = axes[2]
        
        # Create a dense range of thresholds from 0 to 1
        threshold_range = np.linspace(0, 1, 200)
        far_list = [] # False Live Rate
        frr_list = [] # False Fake Rate
        
        # Pre-calculate masks to speed up loop
        fakes = all_probs[fake_indices]
        lives = all_probs[real_indices]
        
        for t in threshold_range:
            # FAR (False Live): % of Fakes with Score >= t
            curr_far = np.mean(fakes >= t) if len(fakes) > 0 else 0.0
            
            # FRR (False Fake): % of Lives with Score < t
            curr_frr = np.mean(lives < t) if len(lives) > 0 else 0.0
            
            far_list.append(curr_far)
            frr_list.append(curr_frr)
            
        ax_err.plot(threshold_range, far_list, color='red', lw=2, label='FAR (False Live)')
        ax_err.plot(threshold_range, frr_list, color='blue', lw=2, label='FRR (False Fake)')
        ax_err.axvline(threshold, color='black', linestyle='--', label=f'Current Tf={threshold:.3f}')
        
        ax_err.set_title("FAR & FRR vs Threshold", fontweight='bold')
        ax_err.set_xlabel('Threshold')
        ax_err.set_ylabel('Error Rate')
        ax_err.set_xlim(0, 1.0)
        ax_err.set_ylim(0, 1.0)
        ax_err.legend()
        ax_err.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        
    # 4. Print Statistics
    print("-" * 40)
    print(f"LIVENESS STATS (Tf={threshold:.4f})")
    print("-" * 40)
    print(f"Accuracy:          {acc*100:.3f}%")
    print(f"False Live Rate:   {flr*100:.3f}%") 
    print(f"False Fake Rate:   {ffr*100:.3f}%")
    print(f"Average Error:     {acer*100:.3f}%")
    print("-" * 40)
    
    return all_probs, all_labels, flr, ffr





def download_liveness_test_paths(data_dir):
    """
    Loads a specific Liveness Test Set from a directory.
    Structure Expected:
       data_dir/
          fake/ (img1.jpg, img2.jpg ...) -> Label 0
          real/ (img1.jpg, img2.jpg ...) -> Label 1
    Args:
        data_dir (str): Path to the folder containing 'real' and 'fake' subfolders.

    Returns:
        test_data (Bunch): Object with .images (paths) and .target (labels).
    """
    classes = {'fake': 0, 'real': 1}
    target_names = np.array(['Fake', 'Real'])
    
    all_paths = [] 
    all_labels = []

    # print(f"Scanning External Test Set at: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return Bunch(images=np.array([]), target=np.array([]), target_names=target_names)

    # Iterate through 'real' and 'fake' folders
    for class_name, label_id in classes.items():
        class_path = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_path):
            print(f"Warning: Folder '{class_name}' not found in {data_dir}")
            continue
            
        # Walk through files (supports subfolders if any, or flat structure)
        for root, dirs, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(root, file)
                    
                    all_paths.append(img_path)
                    all_labels.append(label_id)

    # Convert to Numpy Arrays
    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)

    print(f"Found {len(all_paths)} images.")
    print(f" - Real: {np.sum(all_labels == 1)}")
    print(f" - Fake: {np.sum(all_labels == 0)}")

    # Return Bunch
    return Bunch(images=all_paths, target=all_labels, target_names=target_names)




def filter_non_live_faces(original_data, test_data_paths, loader, model, device, threshold=0.5, verbose=False):
    """
    Detects liveness for all faces in the loader, filters out the 'Fake' ones from BOTH 
    the detected faces dataset and the original paths dataset.
    
    Args:
        original_data (Bunch): The object containing detected faces (.images as numpy array) and .target.
        test_data_paths (Bunch): The object containing original file paths (.images as strings).
        loader (DataLoader): Loader corresponding exactly to original_data (shuffle=False).
        model: Trained liveness model.
        device: 'cuda' or 'cpu'.
        threshold: Score above which a face is considered live.
        verbose: Whether to print detailed logs.
        
    Returns:
        tuple: (filtered_detected_bunch, filtered_paths_bunch) containing only live faces.
    """
    if verbose: print(f"\nRunning Liveness Filtering (Threshold={threshold})...")
    model.eval()
    model.to(device)
    
    all_probs = []

    # 1. Robust Type Check
    is_bunch = hasattr(original_data, 'images') and hasattr(test_data_paths, 'images')

    if is_bunch:
        total_images = len(original_data.images)
        # Sanity check: Ensure both arrays have the same length
        if len(test_data_paths.images) != total_images:
            raise ValueError(f"Mismatch! Detected faces ({total_images}) vs Paths ({len(test_data_paths.images)})")

    if verbose: print(f" Analyzing {total_images} faces...")
    # 2. Get Scores
    with torch.no_grad():
        for batch in loader:
            # Handle Loader formats: (images, targets) OR just (images)
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().view(-1).numpy()
            all_probs.extend(probs)
    
    all_probs = np.array(all_probs)
    if verbose: print(f" -> Liveness Probabilities: {all_probs}")
    
    # 3. Create Mask
    live_mask = all_probs >= threshold
    n_kept = np.sum(live_mask)
    n_removed = total_images - n_kept
    
    if verbose: print(f" -> Total Faces:   {total_images}")
    if verbose: print(f" -> Fakes Removed: {n_removed}")
    if verbose: print(f" -> Lives Kept:    {n_kept}")
    
    # Handle case where everything is rejected
    if n_kept == 0:
        print("WARNING: All faces were classified as Fake!")
        empty_bunch_det = Bunch(images=np.array([]), target=np.array([]), target_names=original_data.target_names)
        empty_bunch_path = Bunch(images=np.array([]), target=np.array([]), target_names=test_data_paths.target_names)
        return empty_bunch_det, empty_bunch_path

    # 4. Apply Mask correctly to BOTH datasets
    # A. Process Detected Faces (original_data)
    # Permute dimensions: (C, H, W) -> (H, W, C) for display/saving usage later
    for i in range(len(original_data.images)):
        if original_data.images[i].shape[0] == 3: # Check if Channels-first
            original_data.images[i] = np.transpose(original_data.images[i], (1, 2, 0))
            
    filtered_detected_faces = Bunch(
        images=original_data.images[live_mask],
        target=original_data.target[live_mask],
        target_names=original_data.target_names
    )
    
    # B. Process File Paths (test_data_paths)
    filtered_file_paths = Bunch(
        images=test_data_paths.images[live_mask],
        target=test_data_paths.target[live_mask],
        target_names=test_data_paths.target_names
    )
    
    return filtered_detected_faces, filtered_file_paths
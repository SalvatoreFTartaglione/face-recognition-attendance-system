import random
import shutil
import numpy as np
import pandas as pd
import cv2
import torch
import os
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from skimage.transform import resize
from sklearn.utils import Bunch
from sklearn.utils import shuffle
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import time
from deepface import DeepFace
from facenet_pytorch import InceptionResnetV1
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import cdist




def download_dataset(data_dir):
    """
    Scans a local dataset with specific 'galleries' and 'probes' folder structure.
    Returns Bunch objects containing FILE PATHS (strings) instead of loaded images.
    
    Structure Expected:
    data_dir/
       galleries/
          subject_00/
             img.jpg
       probes/
          subject_00/
             img.jpg

    Args:
        data_dir (str): Path to the root folder.
        img_size (tuple): Kept for API compatibility
    Returns:
        tuple: (train_data, test_data, target_names)
               Each *_data is a Bunch with .images (N,) array of strings and .target (N,)
    """

    # 1. Initialize lists
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    valid_subject_names = []
    split_report = [] 

    # 2. Locate folders
    galleries_path = os.path.join(data_dir, "galleries")
    probes_path = os.path.join(data_dir, "probes")
    
    if not os.path.exists(galleries_path):
        raise FileNotFoundError(f"Could not find 'galleries' folder inside {data_dir}")

    # 3. Identify Subjects
    gal_subjects = sorted([d for d in os.listdir(galleries_path) if os.path.isdir(os.path.join(galleries_path, d))])
    print(f"Found {len(gal_subjects)} subjects in 'galleries'. Scanning for paths...")

    # 4. Process each subject
    label_id = 0 
    
    for subject_name in gal_subjects:
        subject_dir = os.path.join(galleries_path, subject_name)
        
        if subject_name.endswith('X'):
            print(f"Skipping subject: {subject_name}")
            continue

        # --- A. Collect Gallery Paths (TRAIN) ---
        gal_path_data = [] # List of tuples: (path, label)
        
        for root, dirs, files in os.walk(subject_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    gal_path_data.append(img_path)

        if len(gal_path_data) < 1:
            print(f"Skipping {subject_name} (Gallery): No images found.")
            continue

        valid_subject_names.append(subject_name)
        
        # --- B. Train Split ---
        # Shuffle paths
        gal_path_data = shuffle(gal_path_data, random_state=42)
        
        # Add to main lists
        for path in gal_path_data:
            train_paths.append(path)
            train_labels.append(label_id)
            
            split_report.append({
                "Subject": subject_name, "Label": label_id, 
                "Path": path, "Split": "Training"
            })

        # --- C. Collect Probe Paths (TEST) ---
        subj_probe_path = os.path.join(probes_path, subject_name)
        if os.path.exists(subj_probe_path):
            for root, dirs, files in os.walk(subj_probe_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(root, file)
                        
                        test_paths.append(img_path)
                        test_labels.append(label_id)
                        
                        split_report.append({
                            "Subject": subject_name, "Label": label_id, 
                            "Path": img_path, "Split": "Test/Probe"
                        })
        
        label_id += 1
        
    # 5. Save Split Report
    df_split = pd.DataFrame(split_report)
    df_split.to_csv(os.path.join(data_dir, "dataset_split_report.csv"), index=False)
    print("\nSplit report saved to 'dataset_split_report.csv'")

    # 6. Convert to Bunch
    target_names = np.array(valid_subject_names)
    
    # Note: .images now contains an array of STRINGS (paths), not image pixels
    train_data = Bunch(
        images=np.array(train_paths), 
        target=np.array(train_labels, dtype=np.int64), 
        target_names=target_names
    )
    test_data = Bunch(
        images=np.array(test_paths), 
        target=np.array(test_labels, dtype=np.int64), 
        target_names=target_names
    )
    
    print("\nDataset paths loaded successfully.")
    print(f"  Train Paths: {len(train_paths)}")
    print(f"  Test Paths:  {len(test_paths)}")

    return train_data, test_data, target_names




def download_demo_data(data_dir, verbose=False):
    """
    Scans all images found in data_dir (and subfolders) and stores their FILE PATHS.
    Does not load image data into memory: loads just the image paths
    Returns:
        Bunch: 
            .images (np.array): 1D Array of file path strings.
            .target (np.array): Dummy targets (-1).
    """
    image_paths = []
    
    if verbose: print(f"Scanning directory: {data_dir}")

    if not os.path.exists(data_dir):
        if verbose: print(f"Error: Directory {data_dir} not found.")
        return Bunch(
            images=np.array([]), 
            target=np.array([]), 
            target_names=np.array(['Unknown'])
        )

    count = 0
    # Walk through root and all subdirectories
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                
                # --- CHANGE: Append the PATH string, do not load the image ---
                image_paths.append(img_path)
                count += 1
                
                if count % 100 == 0:
                    if verbose: print(f"Found {count} images...", end="\r")
                    
    dummy_targets = np.full(count, -1, dtype=int)
    print(f"\nSuccessfully found {count} images.")
    
    # Convert to np.array of strings
    # Note:return 'images' as paths to keep compatibility with your Bunch structure
    return Bunch(
        images=np.array(image_paths), 
        target=dummy_targets, 
        target_names=np.array(['Unknown'])
    )




def download_demo_data_paths(data_dir, verbose=False):
    """
    Loads file paths from a directory. 
    Returns a Bunch compatible with 'detect_faces'.
    """
    image_paths = []

    if verbose: print(f"Scanning : {data_dir}")

    if not os.path.exists(data_dir):
        if verbose: print(f"Error: Directory {data_dir} not found.")
        return Bunch(images=np.array([]), target=np.array([]), target_names=np.array([]))

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                image_paths.append(img_path)
    
    # Store as Objects (Strings)
    images_arr = np.array(image_paths, dtype=object)
    
    # Create Dummy Targets (-1)
    dummy_targets = np.full(len(image_paths), -1, dtype=int)
    if verbose: print(f"Found {len(image_paths)} images.")
    
    return Bunch(
        images=images_arr, 
        target=dummy_targets, 
        target_names=np.array(['Unknown'])
    )





def plot_gallery(images, titles, n_row=4, n_col=4):
    """Helper function to plot a gallery of portraits (supports both grayscale and RGB)"""
    n_plots = min(n_row * n_col, len(images), len(titles))
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_plots):
        plt.subplot(n_row, n_col, i + 1)
        img = images[i]
        if img.ndim == 2:
            plt.imshow(img, cmap=plt.cm.gray)
        else:
            plt.imshow(np.clip(img, 0, 1))
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())



def check_dataset(n_people_to_display, dataset, target_names):
    """
    Visualize samples from a dataset where .images contains FILE PATHS.
    """
    unique_labels = np.unique(dataset.target)
    n_people = min(n_people_to_display, len(unique_labels))
    
    for label in unique_labels[:n_people]:
        mask = dataset.target == label
        person_paths = dataset.images[mask] # These are paths now
        
        # Handle naming
        if target_names is not None:
            raw_name = target_names[label] if label < len(target_names) else str(label)
            person_name = raw_name.rsplit(' ', 1)[-1] if isinstance(raw_name, str) else str(raw_name)
        else:
            person_name = f"ID_{label}"
        
        loaded_images = []
        valid_titles = []
        
        # --- LOAD IMAGES FROM PATHS ---
        for i, path in enumerate(person_paths):
            try:
                with Image.open(path) as img:
                    img = img.convert('RGB')
                    loaded_images.append(np.array(img))
                    valid_titles.append(f"{person_name} #{i+1}")
            except Exception as e:
                print(f"Warning: Could not load {path}")

        if len(loaded_images) > 0:
            # Call your plotting helper
            plot_gallery(loaded_images, valid_titles, n_row=1, n_col=len(loaded_images))



def plot_gallery(images, titles, n_row=3, n_col=4):
    """Helper function to plot a grid of images"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(min(len(images), n_row * n_col)):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i]) # Now this receives Numbers, not Strings
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()
            
            
            

def preprocess_and_embed(data, model_name, device, batch_size=32, verbose=False):
    """
    Handles both 'facenet-pytorch' (PyTorch) and 'deepface' (Keras) models.
    Args:
        data (Bunch): Contains .images (np.array of images) and .target
        model_name (str): Name of the model to use.
        device (str): 'cpu' or 'cuda'
        batch_size (int): Batch size for PyTorch models.
        verbose (bool): Whether to print progress.
    """
    embeddings = []
    images = data.images
    total_imgs = len(images)
    if verbose:
        print(f"--- Extracting features using [{model_name}] ---")

    # ==========================================
    # PATH A: PyTorch Models (FaceNet Custom)
    # ==========================================
    if model_name in ["InceptionResnetV1-VGGFace2", "InceptionResnetV1-CASIA"]:
        
        # 1. Load the specific PyTorch model
        if model_name == "InceptionResnetV1-VGGFace2":
            model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        else:
            model = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
            
        if verbose: 
            print(f"   Mode: PyTorch (Batched processing, size=160x160)")
        
        # 2. Your Original Batched Logic
        for i in range(0, total_imgs, batch_size):
            batch = images[i : i + batch_size]
            processed_batch = []
            
            for img in batch:
                # skimage.resize crashes if input is 'object'.
                img = img.astype(np.float32) 
                # ----------------------------------------
                # Resize to 160x160 for InceptionResnet
                img_resized = resize(img, (160, 160), anti_aliasing=True)
                img_resized = img_resized.astype(np.float32)
                # Normalize [-1, 1]
                img_resized = img_resized * 2.0 - 1.0 
                if img_resized.ndim == 2:
                    img_rgb = np.stack([img_resized, img_resized, img_resized], axis=0)
                else:
                    # HWC -> CHW for PyTorch
                    img_rgb = np.transpose(img_resized, (2, 0, 1))
                processed_batch.append(img_rgb)
            
            batch_tensor = torch.tensor(np.array(processed_batch), dtype=torch.float32).to(device)
            
            with torch.no_grad():
                batch_embeddings = model(batch_tensor)
            embeddings.append(batch_embeddings.detach().cpu().numpy())
            
        # Clean up GPU memory for PyTorch
        del model
        torch.cuda.empty_cache()
        return np.concatenate(embeddings, axis=0)

    # ==========================================
    # PATH B: DeepFace Models
    # ==========================================
    else:
        # Build model once for speed
        model_obj = DeepFace.build_model(model_name)
        if model_obj is None:
            raise ValueError(f"DeepFace.build_model('{model_name}'). Model not available or not supported.")
        
        deepface_embeddings = []
        for i, img in enumerate(images):
            # Preprocessing: Ensure uint8 (0-255) for DeepFace
            
            if img.max() <= 1.0:
                img_uint8 = (img * 255).astype(np.uint8)
            else:
                img_uint8 = img.astype(np.uint8)
            # DeepFace handles resizing automatically (e.g., 112x112 for ArcFace)
            results = DeepFace.represent(
                img_path=img_uint8,
                model_name=model_name,
                # model=model_obj,
                enforce_detection=False,
                detector_backend="skip",
                align=True,
            )
            deepface_embeddings.append(results[0]["embedding"])
            
            if verbose and i % 100 == 0:
                print(f"   Processed {i}/{total_imgs}", end="\r")
            
        return np.array(deepface_embeddings)




def title(y_pred_test, y_test, target_names, i):
    pred_name = target_names[y_pred_test[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)   




def plot_losses(best_mlp):
    if hasattr(best_mlp, 'validation_scores_'):
        fig, ax1 = plt.subplots(figsize=(10, 6))
        color = 'tab:blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Training Loss (Error Value)', color=color)
        ax1.plot(best_mlp.loss_curve_, color=color, label='Training Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Validation Accuracy (0.0 - 1.0)', color=color)
        ax2.plot(best_mlp.validation_scores_, color=color, label='Validation Score', linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        plt.title('Training Loss (Linear) vs. Validation Accuracy')
        fig.tight_layout() 
        plt.grid(True)
        plt.show()
    else:
        print("No validation scores available. Ensure 'early_stopping=True' is set.")
        



def visualize_detected_faces(processed_face_dataset, n_cols=10, verbose=False, images=False):
    """
    Visualizes all detected faces in a grid format with subject info printed.
    Args:
        processed_face_dataset: Bunch with .images and .target
        n_cols: Number of columns in the grid
    """
    if len(processed_face_dataset.images) > 0:
        if verbose: print("\nDisplaying detected faces...")
        n_kept = len(processed_face_dataset.images)
        n_rows = int(np.ceil(n_kept / n_cols))
        
        plt.figure(figsize=(2 * n_cols, 2.8 * n_rows)) # Increased height slightly for titles
        
        for k in range(n_kept):
            plt.subplot(n_rows, n_cols, k + 1)
            img_data = processed_face_dataset.images[k]
            t_id = processed_face_dataset.target[k]
            # 1. Determine Label Name (Fake/Real)
            try:
                label_name = processed_face_dataset.target_names[t_id]
                # Clean up string if needed
                if isinstance(label_name, str):
                    label_name = label_name.rsplit(' ', 1)[-1]
            except:
                label_name = str(t_id)
            
            if verbose: print(f"Image {k}: Subject='{label_name}' (ID: {t_id})")

            # --- Normalize image shape and type for matplotlib ---
            try:
                if torch.is_tensor(img_data):
                    img_data = img_data.detach().cpu().numpy()
            except Exception:
                pass
            
            img_np = np.asarray(img_data)
            # Channels-first (C,H,W) -> (H,W,C)
            if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
                img_np = np.transpose(img_np, (1, 2, 0))

            if img_np.ndim == 3 and img_np.shape[2] == 1:
                img_np = img_np.squeeze(axis=2)

            if np.issubdtype(img_np.dtype, np.floating):
                if img_np.max() > 1.0:
                    img_np = img_np / 255.0
                img_np = np.clip(img_np, 0.0, 1.0)
            
            if img_np.ndim == 2:
                plt.imshow(img_np, cmap='gray')
            else:
                plt.imshow(img_np)
            
            # Title: Class Name
            plt.title(f"{k}: {label_name}", fontsize=8)
            plt.axis('off')
            
        plt.tight_layout()
        if images:
            plt.suptitle(f"Images Detected ({n_kept} total)", fontsize=14, y=1.02)
        else:
            plt.suptitle(f"Faces Detected ({n_kept} total)", fontsize=14, y=1.02)
        plt.show()
    else:
        print("No faces were detected to display.")




        
def create_gallery(features, labels, target_names, filename="gallery.npy", database_dir="./database/gallery", verbose=False):
    """
    Saves features, labels, and names into a SINGLE .npy file (containing a dictionary).
    
    Args:
        features (np.array): Shape (N_samples, 512).
        labels (np.array): Shape (N_samples,).
        target_names (np.array): Names corresponding to IDs.
        filename (str): Name of the file (e.g., "gallery.npy").
        database_dir (str): Directory where the file will be saved.
        verbose (bool): Whether to print detailed information.
    """
    # 1. Create Directory if it doesn't exist
    if not os.path.exists(database_dir):
        print(f"Directory '{database_dir}' not found. Creating it...")
        os.makedirs(database_dir)
        
    full_path = os.path.join(database_dir, filename)
    if verbose: print(f"Creating Gallery at '{full_path}'...")
    
    # 2. Pack data into a dictionary
    gallery_data = {
        "embeddings": features,
        "labels": labels,
        "names": target_names
    }
    
    # 3. Save the dictionary as a single object inside a .npy file
    np.save(full_path, gallery_data, allow_pickle=True)
    
    if verbose:
        print(f"Success! Saved {len(features)} embeddings.")
        print(f"Gallery Shape: {features.shape}") 
        print(f"File Size: {os.path.getsize(full_path) / 1024:.2f} KB")





def load_gallery(filename="gallery.npy", gallery_path="./database/gallery"):
    """
    Loads the gallery data from a single .npy file.
    
    Returns:
        tuple: (embeddings, labels, names)
    """
    full_path = os.path.join(gallery_path, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Gallery file not found at: {full_path}")
        
    # 1. Load the file
    # allow_pickle=True is required because loading a Dictionary object
    data_object = np.load(full_path, allow_pickle=True)
    
    # 2. Extract the dictionary
    # When saving a dict with np.save, it is wrapped in a 0-d array. 
    data = data_object.item()
    
    # 3. Extract arrays from keys
    embeddings = data['embeddings']
    labels = data['labels']
    names = data['names']
    
    return embeddings, labels, names




def check_gallery_content(filename="gallery.npy", database_dir="./database/gallery", n_samples=20):
    """
    Inspects the content of the monolithic .npy gallery file.
    Args:
        filename (str): Name of the gallery file.
        database_dir (str): Directory where the gallery file is located.
        n_samples (int): Number of sample entries to display.
    """
    full_path = os.path.join(database_dir, filename)
    
    if not os.path.exists(full_path):
        print(f"Error: File '{full_path}' not found.")
        return

    try:
        # Load using the same logic as load_gallery
        data_object = np.load(full_path, allow_pickle=True)
        data = data_object.item()
        
        embeddings = data['embeddings']
        labels = data['labels']
        names = data['names']
        
        print(f"\n--- Checking Gallery File: {filename} ---")
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Labels shape:     {labels.shape}")
        print(f"Names shape:      {names.shape}")
        
        print(f"\nSample Data (First {min(n_samples, len(labels))} entries):")
        for i in range(min(n_samples, len(labels))):
            lbl_id = labels[i]
            # Safety check for name index
            name_str = names[lbl_id] if lbl_id < len(names) else f"ID_{lbl_id}"
            
            # Preview vector
            vec_preview = embeddings[i][:5]
            print(f" Entry {i}: Label={lbl_id} ({name_str}) | Vector[:5]={vec_preview}...")
            
        print("\nIntegrity Check: ", end="")
        if len(embeddings) == len(labels):
            print("PASSED (Embeddings and Labels match).")
        else:
            print(f"FAILED (Mismatch: {len(embeddings)} vs {len(labels)}).")
    
    except Exception as e:
        print(f"Error reading file: {e}")





def plot_normalized_confusion_matrix(y_true, y_pred, target_names, model_name):
    """
    Plots a normalized confusion matrix (percentage based).
    Row X represents the True Subject.
    Col Y represents the Predicted Subject.
    """
    print(f"\n=== NORMALIZED CONFUSION MATRIX: {model_name} ===")
    
    # 1. Compute Confusion Matrix
    # normalize='true' converts counts to percentages (rows sum to 1.0)
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    # 2. Setup Plot
    plt.figure(figsize=(12, 10))
    
    # 3. Create Heatmap
    sns.heatmap(cm, 
                annot=True,         # Show numbers
                fmt='.2f',          # Format as '0.95' instead of '0.95324...'
                cmap='Blues',       # Darker blue = Higher percentage
                xticklabels=target_names, # Use the 19 names here
                yticklabels=target_names) # Use the 19 names here
    
    plt.title(f"Normalized Confusion Matrix - {model_name}\n(Values are % of True Class)", fontsize=14)
    plt.ylabel('True Identity (Ground Truth)', fontsize=12)
    plt.xlabel('Predicted Identity (System Output)', fontsize=12)
    
    # Rotate axis labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()




def plot_subject_average_similarity(sim_matrix, probe_labels, gallery_labels, target_names):
    """
    Computes and plots the Average Cosine Similarity between subjects.
    
    Args:
        sim_matrix: The raw (N_probes x M_gallery) similarity matrix.
        probe_labels: Array of IDs for the probes (length N_probes).
        gallery_labels: Array of IDs for the gallery (length M_gallery).
        target_names: List of subject names.
    """
    unique_subjects = sorted(list(set(probe_labels) | set(gallery_labels)))
    n_subjects = len(unique_subjects)
    
    # Initialize the averaged matrix (Subject x Subject)
    avg_sim_matrix = np.zeros((n_subjects, n_subjects))

    print("Computing average similarity per subject...")
    
    # Iterate through every subject pair
    for i, subj_row in enumerate(unique_subjects):      # Probe Subject
        for j, subj_col in enumerate(unique_subjects):  # Gallery Subject
            
            # 1. Find rows in sim_matrix belonging to 'subj_row'
            row_indices = np.where(probe_labels == subj_row)[0]
            
            # 2. Find cols in sim_matrix belonging to 'subj_col'
            col_indices = np.where(gallery_labels == subj_col)[0]
            
            # 3. Extract the sub-block of scores and take the Mean
            if len(row_indices) > 0 and len(col_indices) > 0:
                # np.ix_ creates a mesh to select the specific rectangular block
                sub_block = sim_matrix[np.ix_(row_indices, col_indices)]
                avg_sim_matrix[i, j] = np.mean(sub_block)
            else:
                avg_sim_matrix[i, j] = 0.0

    # Plotting
    plt.figure(figsize=(12, 10))
    
    # Use names if available
    if target_names is not None:
        tick_labels = [target_names[i] for i in unique_subjects]
    else:
        tick_labels = unique_subjects

    sns.heatmap(avg_sim_matrix, 
                annot=True, 
                fmt='.2f',      # Show 2 decimal places
                cmap='viridis', 
                vmin=-1, vmax=1,
                xticklabels=tick_labels, 
                yticklabels=tick_labels)
    
    plt.title("Average Cosine Similarity (Subject vs Subject)")
    plt.xlabel("Gallery Subject")
    plt.ylabel("Probe Subject")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()



def evaluate_rank_n_performance(gallery_embs, gallery_labels, probe_embs, probe_labels, threshold, rank=1, gallery_names=None):
    """
    Calculates Identification Rate (Accuracy), FRR, and FAR at a specific Rank N.
    
    Definitions for this context (Genuine Probes):
      - Accuracy: Correct Identity is in Top-N AND Score >= Threshold.
      - FRR (False Rejection): Correct Identity NOT returned (Score < Threshold or Not in Top-N).
      - FAR (False Acceptance): Correct Identity NOT in Top-N, but a WRONG Identity has Score >= Threshold.
    
    Args:
        gallery_embs (np.array): Shape (M_gallery, D)
        gallery_labels (np.array): Shape (M_gallery,)
        probe_embs (np.array): Shape (N_probes, D)
        probe_labels (np.array): Shape (N_probes,)
        threshold (float): Similarity threshold for acceptance.
        rank (int): Rank N for evaluation.
        gallery_names (list): Optional list of subject names for display.
        
    Returns:
        tuple: (accuracy, frr, far)
    """
    # 1. Compute Similarity Matrix
    sim_matrix = np.dot(probe_embs, gallery_embs.T)
    
    total = len(probe_embs)
    correct_identifications = 0
    false_rejections = 0
    false_acceptances = 0 # Misidentifications with high confidence
    
    print(f"\n{'='*20} PERFORMANCE REPORT (Rank-{rank}) {'='*20}")

    for i in range(total):
        true_id = probe_labels[i]
        
        # Get Name (for display)
        if gallery_names is not None and true_id < len(gallery_names):
            true_name = gallery_names[true_id]
        else:
            true_name = f"ID_{true_id}"

        # 2. Get Top Matches
        sorted_indices = np.argsort(sim_matrix[i])[::-1]
        
        # Top N candidates 
        top_n_indices = sorted_indices[:rank]
        top_n_labels = gallery_labels[top_n_indices]
        top_n_scores = sim_matrix[i][top_n_indices]
        
        # Best Match (Rank-1) specifically for reporting False Accept details
        best_match_idx = sorted_indices[0]
        best_match_label = gallery_labels[best_match_idx]
        best_match_score = sim_matrix[i][best_match_idx]
        
        if gallery_names is not None and best_match_label < len(gallery_names):
            best_match_name = gallery_names[best_match_label]
        else:
            best_match_name = f"ID_{best_match_label}"

        # 3. Check Logic
        if true_id in top_n_labels:
            # --- Scenario A: Correct person IS in the top N ---
            match_idx_in_top = np.where(top_n_labels == true_id)[0][0]
            match_score = top_n_scores[match_idx_in_top]
            
            if match_score >= threshold:
                correct_identifications += 1
            else:
                # FAILURE TYPE: Correct person found, but score too low
                false_rejections += 1
                print(f"[FALSE REJECT] Probe {i} ({true_name}) found at Rank-{match_idx_in_top+1} "
                      f"but Score {match_score:.4f} < Threshold {threshold}")
        else:
            # --- Scenario B: Correct person NOT in Top N ---
            # We now check if we accidentally accepted the WRONG person (FAR)
            
            # Use the highest score among the Top N to see if the system would trigger an accept
            # (Usually we check the absolute Top-1 score for acceptance)
            max_score_in_top_n = np.max(top_n_scores)
            
            if max_score_in_top_n >= threshold:
                # FAILURE TYPE: System confidently matched the wrong person
                false_acceptances += 1
                print(f"[FALSE ACCEPT] Probe {i} ({true_name}). "
                      f"Mistaken for {best_match_name} with Score {best_match_score:.4f}")
            else:
                # FAILURE TYPE: System found wrong person, but correctly rejected them due to low score.
                # However, since it failed to find the RIGHT person, this counts as a False Rejection (Miss).
                false_rejections += 1
                # Optional: Print this case if verbose
                # print(f"[MISS] Probe {i} ({true_name}) not in Top-{rank} and Score low.")

    accuracy = correct_identifications / total
    frr = false_rejections / total
    far = false_acceptances / total
    
    print(f"{'-'*60}")
    print(f"Rank-{rank} Accuracy: {accuracy*100:.2f}%")
    print(f"False Rejection Rate (FRR): {frr*100:.2f}%")
    print(f"False Acceptance Rate (FAR): {far*100:.2f}%")
    print(f"{'='*60}\n")
    
    return accuracy, frr, far



def plot_evaluation_results(evaluation_results):
    """
    Plots a grouped bar chart comparing Cosine vs MLP accuracy.
    Colors are changed from the original reference (Orange/Blue -> Purple/Green).
    Args:
        evaluation_results (dict): Contains keys 'model_name', 'mlp_acc', 'cos_dis_acc'.
    Returns:
        None
    """
    # 1. Extract Data
    models = evaluation_results["model_name"]
    mlp_acc = evaluation_results["mlp_acc"]
    cos_acc = evaluation_results["cos_dis_acc"]
    
    # 2. Setup Plot Dimensions
    x = np.arange(len(models))  # Label locations
    width = 0.35                # Width of the bars
    
    plt.figure(figsize=(18, 9)) # Wide layout like the image
    
    # 3. Create Bars (Cosine on left, MLP on right)
    plt.bar(x - width/2, cos_acc, width, label='Cosine', color="#0029f5", edgecolor='black', alpha=0.9)
    plt.bar(x + width/2, mlp_acc, width, label='MLP', color="#ff2f00", edgecolor='black', alpha=0.9)

    # 4. Formatting
    plt.xlabel('Model Architecture', fontsize=12, fontweight='bold')
    plt.ylabel('Rank-1 Accuracy', fontsize=12)
    plt.title('Final Accuracy Comparison: Cosine vs MLP', fontsize=15, fontweight='bold')
    
    # X-Axis Ticks
    plt.xticks(x, models, fontsize=11)
    
    # Y-Axis Scaling (Zoom in like the reference image)
    # find the lowest score and subtract a bit of margin to set the bottom limit
    all_scores = mlp_acc + cos_acc
    if len(all_scores) > 0:
        min_score = min(all_scores)
        plt.ylim(bottom=max(0, min_score - 0.05), top=1.01) # Dynamic zoom
    
    # Add Grid
    plt.grid(axis='y', linestyle='-', alpha=0.3)
    
    # Legend
    plt.legend(title="Classifier Method", loc='lower right')
    
    # Show Plot
    plt.tight_layout()
    plt.show()
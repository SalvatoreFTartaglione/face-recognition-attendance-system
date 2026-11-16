import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from iresnet import iresnet50
import torch
from sklearn.utils import Bunch
from skimage import transform as trans
from facenet_pytorch import MTCNN 
import matplotlib.pyplot as plt
import utilities as utils
from sklearn.preprocessing import normalize


class MagFaceEvaluator:
    def __init__(self, model_path, architecture='iresnet50', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 1. Initialize Architecture
        if architecture == 'iresnet50':
            self.model = iresnet50(pretrained=False, num_classes=512)
        else:
            print("Using iresnet50 architecture for MagFace.")
        
        print(f"Loading MagFace weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 2. Extract State Dict (Unwrap the file structure)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        # 3. AUTO-DETECT PREFIX
        prefix = None
        target_layer = "conv1.weight"
        for k in state_dict.keys():
            if k.endswith(target_layer):
                # Calculate the prefix. 
                # e.g., if k is "module.conv1.weight", prefix is "module."
                prefix = k[: -len(target_layer)]
                # print(f"DEBUG: Detected prefix '{prefix}'")
                break
        
        if prefix is None:
            # Fallback: Print first 5 keys to help debug if detection fails
            print("DEBUG: Could not find 'conv1.weight' in file.")
            print("First 5 keys in file:", list(state_dict.keys())[:5])
            raise RuntimeError("Could not determine model prefix automatically.")

        # 4. Clean Keys using the detected prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                # Remove the prefix
                new_key = k[len(prefix):]
                new_state_dict[new_key] = v
            else:
                # Keep as is (unlikely to match, but safe)
                new_state_dict[k] = v
            
        # 5. Load Weights (Strict=False to ignore the final FC layer mismatch)
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        
        # 6. Verify Critical Layers
        if 'conv1.weight' in missing:
            # This should barely happen now with the smart loader
            raise RuntimeError(f"CRITICAL: 'conv1.weight' is still missing. Keys cleaned to: {list(new_state_dict.keys())[:3]}...")
            
        # print("Success: Backbone weights loaded correctly!")
        
        self.model.to(self.device)
        self.model.eval()

        # 7. Preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])





def compute_image_quality(self, face_image):
    """
    Args:
        face_image: RGB image (numpy array) cropped to the face.
    Returns:
        float: Quality score (Magnitude of the embedding).
    """
    # Preprocess
    input_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
    
    with torch.no_grad():
        embedding = self.model(input_tensor)
        quality_score = torch.norm(embedding, p=2, dim=1).item()
        
    return quality_score
    
    



REFERENCE_LANDMARKS_112 = np.array([
    [38.2946, 51.6963], # Left Eye
    [73.5318, 51.6963], # Right Eye
    [56.0252, 71.7366], # Nose
    [41.5493, 92.3655], # Left Mouth
    [70.7299, 92.3655]  # Right Mouth
], dtype=np.float32)




def align_face_magface_mtcnn(cv2_img, detector):
    """
    Aligns a face in the given image using MTCNN detector for MagFace quality assessment.
    Args:
        cv2_img: Input image in RGB format (numpy array).
        detector: An instance of facenet_pytorch.MTCNN for face detection and landmark extraction.
    Returns:
        aligned_img: Aligned face image (112x112) or None if alignment failed.
        is_aligned: Boolean indicating if alignment was successful.
    """
    # Run detection on the RAW image
    boxes, probs, landmarks = detector.detect(cv2_img, landmarks=True)
    
    # If no face is found in a raw image, it fails to align.
    if landmarks is None or len(landmarks) == 0:
        return None, False  # Return None to signal failure
    
    facial5points = landmarks[0].astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(facial5points, REFERENCE_LANDMARKS_112)
    M = tform.params[0:2, :]
    aligned_img = cv2.warpAffine(cv2_img, M, (112, 112), borderValue=0.0)
    
    return aligned_img, True



def quality_check_magface(magface, dataset, processed_dataset, verbose=False, device='cpu', img_quality_threshold=20):
    """
    Performs quality check on a dataset of FILE PATHS using MagFace, 
    but returns the corresponding PRE-PROCESSED TENSOR from processed_dataset.
    
    Args:
        magface: MagFaceEvaluator instance.
        dataset: Bunch with .images (LIST OF FILE PATHS) and .target.
        processed_dataset: Bunch with .images (TENSORS/ARRAYS) already cropped/detected must correspond 1-to-1 with 'dataset'.
        verbose: Whether to print progress.
        device: Device to run computations on.
        img_quality_threshold: Minimum score to pass.
        
    Returns:
        Bunch: A Bunch containing the single best image TENSOR from processed_dataset.
    """
    
    # Initialize Aligner (Same as in threshold computation)
    mtcnn_aligner = MTCNN(keep_all=False, device=device)
    
    best_tensor = None
    best_target = None
    best_quality = -1.0
    best_idx = -1
    passed_any = False
    n_images = len(dataset.images)
    # Sanity Check
    if len(processed_dataset.images) != n_images:
        raise ValueError(f"Mismatch: Paths ({n_images}) vs Processed Faces ({len(processed_dataset.images)})")

    if verbose: print(f"\n--- MagFace Quality Check (Paths Input) on {n_images} images ---")

    for i in range(n_images):
        img_path = dataset.images[i]
        target = dataset.target[i]
        # Load & Align
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                if verbose: print(f"DROPPED [Load Failed]: {img_path}")
                
            else:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                aligned_img, is_aligned = align_face_magface_mtcnn(img_rgb, mtcnn_aligner)
                if aligned_img is None: 
                    if verbose: print(f"Warning: Could not align {img_path}. Skipping.")
                # 3. COMPUTE QUALITY
                # MagFace requires the aligned 112x112 image
                score = compute_image_quality(magface, aligned_img)
                
                status = "PASS" if score > img_quality_threshold else "FAIL"
                align_status = "Aligned" if is_aligned else "Raw-Resize"
                
                if verbose: print(f"Img {i}: [{align_status}] Score={score:.4f} -> {status}")
                # 4. SELECTION LOGIC
                if score > best_quality:
                    best_quality = score
                    # Select the corresponding TENSOR from the processed_dataset
                    # This grabs the item at index 'i' which corresponds to the file path at index 'i'
                    best_tensor = processed_dataset.images[i] 
                    best_target = target
                    best_idx = i
        except Exception as e:
            if verbose: print(f"Error processing {img_path}: {e}")
            continue
    # Final Reporting
    if best_quality < img_quality_threshold:
        print(f"Warning: Best image found (Score={best_quality:.2f}) is below threshold.")
    if best_tensor is not None:
        print(f"Selection: BEST Image Index {best_idx} (Score={best_quality:.2f})")
    # Return Bunch with the pre-processed tensor (ready for embedding)
    return Bunch(
        images=np.array([best_tensor], dtype=object) if best_tensor is not None else np.array([]),
        target=np.array([best_target]) if best_target is not None else np.array([]),
        target_names=dataset.target_names
    )    
    




def compute_fnmr_at_fmr(embeddings_probe, targets_probe, embeddings_gallery=None, targets_gallery=None, target_fmr=0.001):
    """
    Calculates FNMR at a fixed FMR. Supports both:
    1. Verification (1 Set): All-vs-All comparison (if gallery args are None).
    2. Identification (2 Sets): Probe-vs-Gallery comparison.
    """
    # Ensure embeddings are (N, 512)
    if embeddings_probe.ndim == 3:
        embeddings_probe = embeddings_probe.squeeze(1)
        
    if embeddings_gallery is not None:
        if embeddings_gallery.ndim == 3:
            embeddings_gallery = embeddings_gallery.squeeze(1)

    # --- SCENARIO A: PROBE vs GALLERY (Bipartite) ---
    if embeddings_gallery is not None and targets_gallery is not None:
        # 1. Similarity Matrix (Probe x Gallery)
        sim_matrix = np.dot(embeddings_probe, embeddings_gallery.T)
        
        # 2. Match Matrix
        # Rows = Probes, Cols = Gallery
        match_matrix = (targets_probe[:, None] == targets_gallery[None, :])
        
        # 3. Extract Scores (No diagonal/triu masking needed for bipartite)
        genuine_scores = sim_matrix[match_matrix]
        impostor_scores = sim_matrix[~match_matrix]
        
    # --- SHARED: THRESHOLD & ERROR CALCULATION ---
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return 1.0 # High error if no pairs exist
        
    # Sort impostor scores descending (high similarity = match)
    impostor_scores = np.sort(impostor_scores)[::-1]
    
    # Index for the FMR cutoff
    fmr_index = int(target_fmr * len(impostor_scores))
    threshold = impostor_scores[fmr_index] if fmr_index < len(impostor_scores) else 1.0
    
    # Calculate FNMR
    # FNMR = Ratio of Genuine pairs that fall BELOW the threshold
    false_non_matches = np.sum(genuine_scores < threshold)
    fnmr = false_non_matches / len(genuine_scores)
    
    return fnmr





def merge_quality_and_embeddings(magface, mtcnn_aligner_magface, device, dataset, precomputed_embs, desc, verbose=True):
    """
    Merges MagFace quality scores with pre-computed embeddings for a dataset of FILE PATHS.
    Args:
        magface: MagFaceEvaluator instance.
        mtcnn_aligner_magface: MTCNN instance for alignment.
        device: Device to run computations on.
        dataset: Bunch with .images (LIST OF FILE PATHS) and .target.
        precomputed_embs: (N, D) numpy array of pre-computed embeddings corresponding to dataset.images.
        desc: Description string for logging.
        verbose: Whether to print progress.
    Returns:
        Tuple: (embeddings (N_valid, D), qualities (N_valid,), targets (N_valid,))
    """
    valid_qualities = []
    valid_embeddings = [] # pick these from the pre-computed list
    valid_targets = []
    
    n_images = len(dataset.images)
    
    # Ensure data consistency
    if n_images != len(precomputed_embs):
        raise ValueError(f"Mismatch in {desc}: {n_images} images vs {len(precomputed_embs)} embeddings.")
    
    for i in range(n_images):
        img_path = dataset.images[i]
        target = dataset.target[i]
        existing_embedding = precomputed_embs[i]
        
        try:
            # 1. Load Image (For MagFace Quality Calculation)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: 
                if verbose: print(f"DROPPED [Load Failed]: {img_path}")
                continue
            # 2. MagFace Path (Compute Quality)
            # Convert to RGB -> Align 112x112
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            aligned_112, is_aligned = align_face_magface_mtcnn(img_rgb, mtcnn_aligner_magface)
            if aligned_112 is None: 
                # If MagFace can't find a face, drop this image, even if Inception found it.
                if verbose: print(f"DROPPED [MagFace Align Failed]: {img_path}")
                continue

            # Calculate Quality Score
            input_blob = magface.transform(aligned_112).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = magface.model(input_blob)
                quality = torch.norm(feat, p=2, dim=1).item()
            
            # 3. Embedding Path (Retrieve Pre-computed)
            #  take the embedding that corresponds to this index
            valid_qualities.append(quality)
            valid_embeddings.append(existing_embedding)
            valid_targets.append(target)

            if verbose and len(valid_qualities) % 30 == 0:
                window = min(30, len(valid_qualities))
                avg_quality = float(np.mean(valid_qualities[-window:]))
                print(f"[{desc}] Processing {i+1}/{n_images} - Avg Quality (last {window} samples): {avg_quality:.2f}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print(f"\n[{desc}] Completed. Valid pairs: {len(valid_qualities)}/{n_images}")
    
    return np.array(valid_embeddings), np.array(valid_qualities), np.array(valid_targets)






def compute_magface_threshold(magface, gallery_data, gallery_embeddings, probe_data, probe_embeddings, device, target_fmr=0.001, verbose=True):
    """
    Computes optimal quality threshold using a Hybrid Approach with PRE-COMPUTED Embeddings.
    
    Args:
        magface: MagFaceEvaluator instance.
        gallery_data: Bunch with .images (file paths) and .target.
        gallery_embeddings: (N, D) numpy array of embeddings corresponding to gallery_data.images.
        probe_data: Bunch with .images (file paths) and .target.
        probe_embeddings: (M, D) numpy array of embeddings corresponding to probe_data.images.
        device: Device to run computations on.
        target_fmr: Target FMR for FNMR calculation.
        verbose: Whether to print info and progresses.
    Returns:
        float: Optimal quality threshold.
    """
    
    # Initialize Aligner for MagFace (Strict 112x112)
    mtcnn_aligner_magface = MTCNN(keep_all=False, device=device)
    
    print(f"\n--- Hybrid Threshold Evaluation (Pre-computed Embeddings) ---")
    print(f"Gallery Images: {len(gallery_data.images)} | Embeddings: {gallery_embeddings.shape}")
    print(f"Probe Images:   {len(probe_data.images)} | Embeddings: {probe_embeddings.shape}\n")

    # --- PHASE 1: Collect Data (Merge Quality + Embeddings) ---
    print(">>> Processing Gallery...")
    gal_embs, gal_qualities, gal_targets =  merge_quality_and_embeddings(magface, mtcnn_aligner_magface, device, gallery_data, gallery_embeddings, "Gallery", verbose)
    gal_embs = normalize(gal_embs, norm='l2')

    # FILTER GALLERY: Optional but recommended
    print(f"Filtered Gallery: {len(gal_embs)} images ")
    print("\n>>> Processing Probe...")
    probe_embs, probe_qualities, probe_targets = merge_quality_and_embeddings(magface, mtcnn_aligner_magface, device, probe_data, probe_embeddings, "Probe", verbose)
    probe_embs = normalize(probe_embs, norm='l2')
    n_probes = len(probe_qualities)

    # --- PHASE 2: Simulate Rejection (ERC Curve) ---
    reject_ratios = np.arange(0, 0.51, 0.01)
    error_rates = []
    thresholds_at_ratio = []
    # Sort Probes by MagFace quality
    sorted_indices = np.argsort(probe_qualities)
    print("\n--- Simulating Rejection (Probe Filtering) ---")
    print(f"{'Reject Ratio':<15} | {'Quality Thresh':<15} | {'FNMR (Error)':<15}")
    print("-" * 50)

    for ratio in reject_ratios:
        n_drop = int(n_probes * ratio)
        keep_indices = sorted_indices[n_drop:]
        if n_drop > 0:
            curr_quality_thresh = probe_qualities[sorted_indices[n_drop-1]]
        else:
            curr_quality_thresh = np.min(probe_qualities)
            
        thresholds_at_ratio.append(curr_quality_thresh)
        
        subset_probe_embs = probe_embs[keep_indices]
        subset_probe_targets = probe_targets[keep_indices]
        
        if len(subset_probe_embs) == 0:
            fnmr = 1.0
        else:
            # Bipartite FNMR: Filtered Probes vs Clean Gallery
            fnmr = compute_fnmr_at_fmr(embeddings_probe=subset_probe_embs,
                targets_probe=subset_probe_targets,
                embeddings_gallery=gal_embs,
                targets_gallery=gal_targets,
                target_fmr=target_fmr
            )
            
        error_rates.append(fnmr)
        print(f"{ratio*100:>5.1f}%          | {curr_quality_thresh:>10.4f}      | {fnmr:>10.4%}")

# --- PHASE 3: Permissive Selection with Local Refinement ---
    error_rates_np = np.asarray(error_rates, dtype=float)
    min_error_global = float(np.min(error_rates_np))
    
    # 1. FIND ANCHOR (The "Permissive" Starting Point)
    # This finds the first point where error is decent (< 1% or close to global min)
    # Same logic as before to find that 16% point.
    target_performance = max(0.01, min_error_global * 1.20)
    
    anchor_idx = -1
    hard_limit_idx = np.searchsorted(reject_ratios, 0.30) # Hard limit 30%
    
    for i in range(hard_limit_idx):
        if error_rates_np[i] <= target_performance:
            anchor_idx = i
            break
            
    if anchor_idx == -1:
        # Fallback: Just pick best error in range
        anchor_idx = np.argmin(error_rates_np[:hard_limit_idx])

    # 2. REFINE (Look for Higher Thresholds nearby)
    # We look at the anchor and the next few points (e.g., +10% rejection range)
    # We want to see if we can get a higher quality score with similar error.
    
    # Define "Similar Error": Allow error to vary slightly (e.g., +0.05%) from anchor
    anchor_error = error_rates_np[anchor_idx]
    local_tolerance = anchor_error + 0.05 
    
    # Search range: From Anchor up to Hard Limit (30%)
    search_indices = range(anchor_idx, hard_limit_idx)
    
    candidate_indices = []
    
    for i in search_indices:
        # If error is still good (similar to anchor), it's a candidate
        if error_rates_np[i] <= local_tolerance:
            candidate_indices.append(i)
            
    # 3. SELECT MAX THRESHOLD
    # From the candidates, pick the one with the HIGHEST Quality Threshold
    # (Since array is sorted by quality, higher index = higher threshold)
    best_idx = np.max(candidate_indices) # This picks the right-most (highest quality) candidate
    
    optimal_threshold = thresholds_at_ratio[best_idx]
    optimal_ratio = reject_ratios[best_idx]
    selected_error = error_rates_np[best_idx]

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(reject_ratios * 100, error_rates, marker='o', linestyle='-', linewidth=2, color='blue', label='FNMR Curve')
    
    # Visuals
    plt.axhline(y=min_error_global, color='green', linestyle=':', label=f'Global Min ({min_error_global:.2%})')
    plt.axhline(y=target_performance, color='orange', linestyle='--', label=f'Base Target ({target_performance:.2%})')
    plt.axvline(x=optimal_ratio*100, color='red', linestyle='--', label=f"Refined Cutoff ({optimal_ratio*100:.0f}%)")
    
    # Show the Anchor (Starting Point) vs Selected (Refined)
    plt.scatter([reject_ratios[anchor_idx]*100], [error_rates[anchor_idx]], color='gray', s=50, label='Anchor', zorder=5)
    plt.scatter([optimal_ratio*100], [selected_error], color='red', s=150, label='Selected', zorder=6)
    
    plt.text(optimal_ratio*100 + 1, selected_error, f" Final Thresh: {optimal_threshold:.3f}", color='red', fontweight='bold', fontsize=12)
    
    plt.title(f"ERC Analysis\nSelected: {selected_error:.2%} @ {optimal_threshold:.3f} (Rejected {optimal_ratio:.0%})")
    plt.xlabel("Reject Ratio (%)")
    plt.ylabel("FNMR")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    print(f"="*60)
    print (f"\nOptimal Quality Threshold determined: {optimal_threshold:.4f}\nReject Ratio: {optimal_ratio*100:.2f}%\nFNMR: {selected_error:.4%})")
    
    return optimal_threshold

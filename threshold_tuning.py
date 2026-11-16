"""
Matching (identification / verification) and threshold calibration.

All embeddings are assumed to be L2-normalised, so the dot product equals the
cosine similarity (1 = identical, higher = more similar).
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold


def fnir_curve(probe_embs, probe_labels, gal_embs, gal_labels, thresholds):
    """
    False Negative Identification Rate for every threshold (closed-set probes).

    A probe is a miss when its Rank-1 identity is wrong OR its Rank-1 score is
    below the threshold. Unlike the image-to-image FRR, this is the actual
    error of the 1:N search.
    """
    sim = np.dot(probe_embs, gal_embs.T)
    top_idx = np.argmax(sim, axis=1)
    top_score = sim[np.arange(len(sim)), top_idx]
    correct = np.asarray(gal_labels)[top_idx] == np.asarray(probe_labels)
    hits = correct[None, :] & (top_score[None, :] >= np.asarray(thresholds)[:, None])
    return 1.0 - hits.mean(axis=1)



def compute_similarities(val_embeddings, val_labels, gallery_embeddings, gallery_labels):
    """
    Computes all pairwise Cosine Similarities (1.0 = Match, 0.0 = Different).
    Assumes embeddings are already NORMALIZED.
    Args:
        val_embeddings (np.array): Shape (N_val, 512). Validation set embeddings.
        val_labels (np.array): Shape (N_val,). IDs for the validation set.
        gallery_embeddings (np.array): Shape (N_gal, 512). Gallery set embeddings.
        gallery_labels (np.array): Shape (N_gal,). IDs for the gallery set.
    Returns:
        tuple: (genuine_scores, impostor_scores)
    """
    # print("Computing similarities...")
    
    # Matrix Multiplication: (N_val, 512) x (N_gal, 512).T = (N_val, N_gal)
    # Result: A matrix where cell [i, j] is the similarity between val[i] and gal[j]
    sim_matrix = np.dot(val_embeddings, gallery_embeddings.T)
    genuine_scores = []
    impostor_scores = []
    
    # Extract scores based on labels
    for i, val_id in enumerate(val_labels):
        # Create a boolean mask for the current row: True where gallery ID matches validation ID
        mask_genuine = (gallery_labels == val_id)
        
        # Similarities with SAME person
        gen_scores = sim_matrix[i][mask_genuine]
        genuine_scores.extend(gen_scores)
        # Similarities with DIFFERENT people
        imp_scores = sim_matrix[i][~mask_genuine]
        impostor_scores.extend(imp_scores)
            
    return np.array(genuine_scores), np.array(impostor_scores)



def identify_user(probe_emb, gal_embs, gal_labels, gal_names, threshold=0.5):
    """
    Identifies a user by comparing their embedding against the entire gallery using SIMILARITY.
    
    Args:
        probe_emb (np.array): Shape (512,) or (1, 512). The live face.
        gal_embs (np.array): Shape (N, 512). The gallery database.
        gal_labels (np.array): Shape (N,). IDs for the gallery.
        gal_names (np.array): Names corresponding to IDs.
        threshold (float): Similarity threshold (0 to 1). Score < Threshold = "Unknown".
        
    Returns:
        tuple: (is_recognized (bool), candidate_name (str), max_score (float))
    """
    # Ensure probe is (1, 512)
    if probe_emb.ndim == 1:
        probe_emb = probe_emb.reshape(1, -1)

    # 1. Compute SIMILARITY to ALL gallery samples (Higher = Better)
    # Shape: (1, N_gallery) -> Flatten to (N_gallery,)
    scores = cosine_similarity(probe_emb, gal_embs)[0]
    
    # 2. Find the index of the Maximum Similarity (Best Match)
    best_idx = np.argmax(scores)
    max_score = scores[best_idx]
    
    # 3. Retrieve Identity Info
    predicted_label_id = gal_labels[best_idx]
    
    # Safety check for name index
    if predicted_label_id < len(gal_names):
        candidate_name = gal_names[predicted_label_id]
    else:
        candidate_name = f"ID_{predicted_label_id}"

    # 4. Threshold Check (Open-Set Recognition): accept only if score >= threshold
    return bool(max_score >= threshold), candidate_name, max_score



def rank_candidates(probe_emb, gal_embs, gal_labels, gal_names, threshold):
    """
    Open-set identification output required by the system specification: the
    list of enrolled users whose score is above the threshold, ordered by
    matching score (best first). Each identity is scored with its best template.

    Returns:
        list of dict: [{"id", "name", "score"}, ...] (empty list = reject / unknown user)
    """
    scores = cosine_similarity(np.asarray(probe_emb).reshape(1, -1), gal_embs)[0]
    candidates = []
    for label in np.unique(gal_labels):
        best = float(np.max(scores[gal_labels == label]))
        if best >= threshold:
            name = gal_names[label] if label < len(gal_names) else f"ID_{label}"
            candidates.append({"id": int(label), "name": str(name), "score": best})
    return sorted(candidates, key=lambda c: c["score"], reverse=True)




def verify_user(claimed_id, probe_embedding, gallery_embeddings, gallery_labels, threshold):
    """
    User Verification (1:1)
    Matches the probe ONLY against the gallery templates of the claimed identity.
    Args:
        claimed_id (int): The ID the user claims to be.
        probe_embedding (np.array): Shape (512,). The live face embedding.
        gallery_embeddings (np.array): Shape (N, 512). The gallery database.
        gallery_labels (np.array): Shape (N,). IDs for the gallery.
        threshold (float): Similarity threshold Tv. Score >= Tv -> "MATCH".
    Returns:
        tuple: (best_score, decision) with decision in {"MATCH", "REJECT"}.
    Raises:
        ValueError: if the claimed ID is not enrolled.
    """
    # 1. Select only the templates belonging to the Claimed ID
    user_indices = np.where(gallery_labels == claimed_id)[0]
    if len(user_indices) == 0:
        raise ValueError(f"Claimed ID {claimed_id} is not enrolled in the gallery.")

    # 2. Similarity against the claimed user's templates, keep the best one
    sim_scores = cosine_similarity(np.asarray(probe_embedding).reshape(1, -1), gallery_embeddings[user_indices])[0]
    best_score = float(np.max(sim_scores))

    # 3. Binary decision (same >= convention used during threshold tuning)
    return best_score, ("MATCH" if best_score >= threshold else "REJECT")
    

    


def compute_scores_for_tuning(val_embeddings, val_labels, gallery_embeddings, gallery_labels):
    """
    Computes scores for BOTH Verification (All-Pairs) and Identification (Max-Impostor).
    Args:
        val_embeddings (np.array): Shape (N_val, 512). Validation set embeddings.
        val_labels (np.array): Shape (N_val,). IDs for the validation set.
        gallery_embeddings (np.array): Shape (N_gal, 512). Gallery set embeddings.
        gallery_labels (np.array): Shape (N_gal,). IDs for the gallery set.
    Returns:
        tuple: (genuine_scores, all_impostor_scores, max_impostor_scores)
    """
    # Matrix Multiplication
    sim_matrix = np.dot(val_embeddings, gallery_embeddings.T)
    
    genuine_scores = []
    all_impostor_scores = [] # For Verification (Standard FAR)
    max_impostor_scores = [] # For Identification (Worst-Case FAR)
    
    for i, val_id in enumerate(val_labels):
        # Mask for the same person (Genuine)
        mask_genuine = (gallery_labels == val_id)
        # 1. Genuine Scores
        gen_scores = sim_matrix[i][mask_genuine]
        genuine_scores.extend(gen_scores)
        # 2. Impostor Scores
        imp_scores = sim_matrix[i][~mask_genuine]
        if len(imp_scores) > 0:
            # A. For Verification: Keep ALL scores (Image-to-Image)
            all_impostor_scores.extend(imp_scores)
            # B. For Identification: Keep only MAX score (Open-Set)
            max_impostor_scores.append(np.max(imp_scores))
            
    return np.array(genuine_scores), np.array(all_impostor_scores), np.array(max_impostor_scores)



def compute_liveness_threshold_kfold(labels, probs, k=4):
    """
    Computes the decision threshold using Stratified K-Fold Cross Validation.
    
    Plots 3 charts per fold:
    1. Score Distribution (Validation Set)
    2. ROC Curve (Training Set)
    3. FAR & FRR vs Threshold (Training Set)
    Args:
        labels (np.array): True binary labels (1=Live, 0=Fake).
        probs (np.array): Predicted liveness probabilities.
        k (int): Number of folds for cross-validation.
    Returns:
        float: Average optimal threshold across folds.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_thresholds = []
    
    print(f"\n{'='*60}")
    print(f"RUNNING STRATIFIED {k}-FOLD CROSS-VALIDATION")
    print(f"{'='*60}")

    # Create a large figure to hold all fold plots (3 columns now)
    # Width increased to 18 to accommodate the 3rd plot
    fig, axes = plt.subplots(k, 3, figsize=(18, 5 * k))
    fig.suptitle(f'Cross-Validation Analysis ({k} Folds)', fontsize=16, y=0.92)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Handle case where k=1 to avoid indexing errors
    if k == 1: axes = np.array([axes])

    for fold_idx, (train_index, val_index) in enumerate(skf.split(probs, labels)):
        # 1. Split Data
        X_train, X_val = probs[train_index], probs[val_index]
        y_train, y_val = labels[train_index], labels[val_index]
        
        # 2. Compute ROC on TRAINING Set
        fpr, tpr, thresholds = roc_curve(y_train, X_train)
        roc_auc = auc(fpr, tpr)
        
        # Calculate FRR and FAR for plotting
        # FAR is just FPR
        far_curve = fpr
        # FRR is 1 - TPR
        frr_curve = 1 - tpr
        
        # 3. Find Threshold (on Training Set)
        # Find where FAR and FRR are closest
        eer_index = np.nanargmin(np.abs(far_curve - frr_curve))
        best_threshold = thresholds[eer_index]
        metric_val = far_curve[eer_index]
        metric_name = "EER"
        fold_thresholds.append(best_threshold)
        # 4. Visualization
        ax_hist = axes[fold_idx, 0]
        ax_roc  = axes[fold_idx, 1]
        ax_err  = axes[fold_idx, 2] # New Axis
        
        # --- Plot A: Score Distribution (Validation Set) ---
        # Shows how well the model separates Live vs Fake on unseen data
        val_live = X_val[y_val == 1]
        val_fake = X_val[y_val == 0]
        
        # Safety check for empty bins to prevent warnings
        if len(val_live) > 0:
            ax_hist.hist(val_live, bins=30, alpha=0.5, color='green', label='Live (Val)', density=True)
        if len(val_fake) > 0:
            ax_hist.hist(val_fake, bins=30, alpha=0.5, color='red', label='Fake (Val)', density=True)
            
        ax_hist.axvline(best_threshold, color='blue', linestyle='--', linewidth=2, label=f'Tf={best_threshold:.3f}')
        ax_hist.set_title(f"Fold {fold_idx+1}: Val Scores", fontweight='bold')
        ax_hist.set_xlabel("Liveness Probability")
        ax_hist.legend()
        ax_hist.grid(alpha=0.3)

        # --- Plot B: ROC Curve (Training Set) ---
        # Standard ROC visualization
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        
        # Mark selected threshold on ROC
        # find the index in the ROC thresholds closest to our chosen best_threshold
        if len(thresholds) > 0:
            closest_idx = np.argmin(np.abs(thresholds - best_threshold))
            ax_roc.scatter(fpr[closest_idx], tpr[closest_idx], color='black', zorder=5, label='Selected Tf')
            
        ax_roc.set_title(f"Fold {fold_idx+1}: ROC Curve", fontweight='bold')
        ax_roc.set_xlabel('False Accept Rate (FAR)')
        ax_roc.set_ylabel('True Accept Rate (TPR)')
        ax_roc.legend()
        ax_roc.grid(alpha=0.3)
        
        # --- Plot C: FAR vs FRR (Training Set) ---
        # This ensures the graph looks like a smooth X shape
        t_dense = np.linspace(0, 1, 200)
        far_dense = []
        frr_dense = []
        

        train_live = X_train[y_train == 1]
        train_fake = X_train[y_train == 0]
        
        for t in t_dense:
            # FAR: % of Fakes classified as Live (Score >= t)
            curr_far = np.mean(train_fake >= t) if len(train_fake) > 0 else 0.0
            # FRR: % of Lives classified as Fake (Score < t)
            curr_frr = np.mean(train_live < t) if len(train_live) > 0 else 0.0
            
            far_dense.append(curr_far)
            frr_dense.append(curr_frr)

        ax_err.plot(t_dense, far_dense, color='red', lw=2, label='FAR (False Accept)')
        ax_err.plot(t_dense, frr_dense, color='blue', lw=2, label='FRR (False Reject)')
        ax_err.axvline(best_threshold, color='black', linestyle='--', label=f'Tf={best_threshold:.3f}')
        
        # Mark the intersection (approximate EER point on this specific fold)
        # plot the metric value at the selected threshold
        ax_err.plot(best_threshold, metric_val, 'ko', label='Selected Point')
        ax_err.set_title(f"Fold {fold_idx+1}: FAR & FRR vs Threshold", fontweight='bold')
        ax_err.set_xlabel('Threshold')
        ax_err.set_ylabel('Error Rate')
        ax_err.set_xlim(0, 1.0)
        ax_err.set_ylim(0, 1.0) 
        ax_err.legend()
        ax_err.grid(alpha=0.3)
        print(f"Fold {fold_idx+1} | Optimal Threshold: {best_threshold:.4f} | {metric_name}: {metric_val:.4f} | Train AUC: {roc_auc:.4f}")
    plt.show()
    
    # 5. Average Threshold
    avg_threshold = np.mean(fold_thresholds)
    std_threshold = np.std(fold_thresholds)
    print(f"\n{'='*60}")
    print(f"FINAL RESULT ({k}-Fold CV)")
    print(f"{'='*60}")
    print(f"Thresholds found: {[f'{t:.4f}' for t in fold_thresholds]}")
    print(f"Average Threshold: {avg_threshold:.4f} (+/- {std_threshold:.4f})")
    
    return avg_threshold




def compute_far_frr_verification_on_probes(threshold, probe_embs_norm, probe_labels, gal_embs, gal_labels, verbose=True):
    """
    Computes verification FAR and FRR with an Image-to-Image (all-pairs) protocol:
    every probe is compared with every gallery template; same-identity pairs are
    genuine attempts, different-identity pairs are impostor attempts.
    Args:
        threshold (float): The Verification Threshold (Tv). Score >= Tv -> accept.
        probe_embs_norm (np.array): Normalized probe embeddings.
        probe_labels (np.array): Labels for the probe embeddings.
        gal_embs (np.array): Gallery embeddings.
        gal_labels (np.array): Labels for the gallery embeddings.
        verbose (bool): Print a summary.
    Returns:
        tuple: (FAR, FRR)
    """
    sim_matrix = np.dot(probe_embs_norm, gal_embs.T)                      # (N_probes, N_gallery)
    genuine_mask = np.asarray(probe_labels)[:, None] == np.asarray(gal_labels)[None, :]

    genuine_scores = sim_matrix[genuine_mask]
    impostor_scores = sim_matrix[~genuine_mask]
    genuine_attempts, impostor_attempts = len(genuine_scores), len(impostor_scores)
    false_rejects = int(np.sum(genuine_scores < threshold))
    false_accepts = int(np.sum(impostor_scores >= threshold))

    far = false_accepts / impostor_attempts if impostor_attempts > 0 else 0.0
    frr = false_rejects / genuine_attempts if genuine_attempts > 0 else 0.0

    if verbose:
        print(f"{'='*50}")
        print(f"CALCULATING FAR & FRR")
        print(f"Tv = {threshold:.4f}")
        print(f"{'-'*50}")
        print(f"Genuine Comparisons:  {genuine_attempts}")
        print(f"False Rejects:        {false_rejects}")
        print(f"--> FRR:              {frr * 100:.4f}%")
        print(f"{'-'*50}")
        print(f"Impostor Comparisons: {impostor_attempts}")
        print(f"False Accepts:        {false_accepts}")
        print(f"--> FAR:              {far * 100:.4f}%")
        print(f"{'='*50}")
            
    return far, frr




def compute_fnir_identification_on_probes(ident_threshold, best_probe_embs_norm, probe_labels, gal_embs, gal_labels, gal_names):
    """
    Computes Identification metrics (DIR and FNIR) on the full dataset given an identification threshold.
    Adapted to work with the single-output identify_user function.
    Args:
        ident_threshold (float): The Identification Threshold (Ti).
        best_probe_embs_norm (np.array): Normalized probe embeddings.
        probe_labels (np.array): Labels for the probe embeddings.
        gal_embs (np.array): Gallery embeddings.
        gal_labels (np.array): Labels for the gallery embeddings.
        gal_names (list/array): List of subject names for logging.
    Returns:
        float: The calculated FNIR (0.0 to 1.0).
    """
    correct_identifications = 0
    false_negatives = 0  # User was in gallery, but system said "Unknown" or matched wrong person

    #  assume all probes in this list ARE enrolled users (Valid IDs)
    total_probes = len(best_probe_embs_norm)

    for i in range(total_probes):
        current_probe = best_probe_embs_norm[i]
        true_id = probe_labels[i]
        
        # 1. Get the True Name for comparison
        # ( compare names because identify_user returns a name, not an ID)
        if true_id < len(gal_names):
            true_name = gal_names[true_id]
        else:
            true_name = f"ID_{true_id}"

        # 2. Run Identification
        # Returns: (identified_name, score)
        result, pred_name, score = identify_user(current_probe, gal_embs, gal_labels, gal_names, ident_threshold)

        #print(f"Probe {i+1}/{total_probes} (True: {true_name}) -> Pred: {pred_name} (Score: {score:.4f})")

        # 3. Check Result
        if result == False:
            # System Rejected the user (Score < Threshold)
            false_negatives += 1
        elif pred_name == true_name:
            # System Identified the user Correctly
            correct_identifications += 1
        else:
            # System Identified the user, but it was the WRONG person (Misidentification)
            false_negatives += 1

    # 4. Metrics
    dir_rate = correct_identifications / total_probes if total_probes > 0 else 0
    fnir = false_negatives / total_probes if total_probes > 0 else 0

    print(f"\n{'='*50}")
    print(f"CALCULATING IDENTIFICATION METRICS (1:N)")
    print(f"Threshold (Ti) = {ident_threshold}")
    print(f"Total Probes: {total_probes}")
    print(f"Correct Identifications (DIR): {correct_identifications}")
    print(f"Misses (Unknown + Wrong ID):   {false_negatives}")
    print(f"--> Identification Rate (DIR): {dir_rate * 100:.2f}%")
    print(f"--> Miss Rate (FNIR):          {fnir * 100:.2f}%")
    print(f"{'='*50}")

    return fnir



def compute_open_set_far_on_probes(threshold, probe_embs, probe_labels, gal_embs, gal_labels, gal_names):
    """
    Computes the False Acceptance Rate (FAR) for Open Set Identification.
    It simulates 'Unknown' users by temporarily removing a subject from the gallery 
    and checking if their probes are still accepted by the system.
    
    Args:
        threshold (float): The Identification Threshold (Ti).
        probe_embs (np.array): Normalized probe embeddings.
        probe_labels (np.array): Labels for the probe embeddings.
        gal_embs (np.array): Gallery embeddings.
        gal_labels (np.array): Labels for the gallery embeddings.
        gal_names (list/array): List of subject names for logging.
        
    Returns:
        float: The calculated FAR (0.0 to 1.0).
    """
    total_attempts = 0
    false_acceptances = 0
    
    # Get list of all subjects present in the PROBES
    unique_probe_subjects = np.unique(probe_labels)
    
    print(f"\n--- Running Open Set Simulation (Ti={threshold:.4f}) ---")

    for subject_id in unique_probe_subjects:
        
        # 1. CREATE OPEN SET CONDITION
        # Remove THIS subject from the gallery to treat them as an "Unknown/Impostor"
        mask_valid_gallery = (gal_labels != subject_id)
        
        # Safety check: If dataset has only 1 subject, can't do this test
        if not np.any(mask_valid_gallery):
            print(f"Skipping Subject ID {subject_id} - Only subject in gallery.")
            continue

        # Create the temporary "Clean" Gallery (without the current subject)
        current_gal_embs = gal_embs[mask_valid_gallery]
        current_gal_labels = gal_labels[mask_valid_gallery]

        # 2. Get Probes for this Subject
        # These probes now represent "Strangers" attempting to enter
        subject_probe_indices = np.where(probe_labels == subject_id)[0]
        subject_probes = probe_embs[subject_probe_indices]
        
        subject_name = gal_names[subject_id] if gal_names is not None else f"ID_{subject_id}"
        # print(f"Testing Intruder: {subject_name} ({len(subject_probes)} probes)...")

        # 3. Test every probe against the Clean Gallery
        # Optimization: Matrix multiplication (N_probes x M_gallery)
        # Result shape: (Num_Probes, Num_Gallery_Entries)
        sim_matrix = np.dot(subject_probes, current_gal_embs.T)
        
        # For each probe, find the best match in the gallery
        # axis=1 finds the max score across all gallery images for that probe
        best_scores = np.max(sim_matrix, axis=1)
        
        # 4. Check results
        for score in best_scores:
            total_attempts += 1
            if score >= threshold:
                false_acceptances += 1
                
    # 5. Final Calculation
    far = false_acceptances / total_attempts if total_attempts > 0 else 0.0
    
    print(f"{'='*50}")
    print(f"Ti={threshold:.4f}:")
    print(f"  Total Impostor Attempts: {total_attempts}")
    print(f"  False Acceptances:       {false_acceptances}")
    print(f"  --> Open Set FAR:        {far * 100:.4f}%")
    
    return far





def find_optimal_thresholds_kfold(embeddings, labels, k_folds=5, step=0.001, desired_far=0.001, verbose=True, gal_names=None, bias_tv=0.0, bias_ti=0.0):
    """
    Calibrates Tv and Ti with Stratified K-Fold cross-validation on the GALLERY
    (image-disjoint folds: K-1 folds act as gallery, 1 fold as validation probes).

    For each fold:
      - Tv = min{t : FAR_all-impostor(t) <= desired_far} + bias_tv   (1:1 verification)
      - Ti = min{t : FAR_max-impostor(t) <= desired_far} + bias_ti   (1:N open-set identification)
    The final thresholds are the averages across folds.

    Returns:
        dict with keys 'Tv', 'Ti', 'std_Tv', 'std_Ti', 'fold_Tv', 'fold_Ti' and
        'metrics' (mean of the per-fold verification/identification error rates).
        Named keys are used on purpose: a positional tuple made it easy to swap
        the two standard deviations when unpacking.
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    tv_scores = []
    ti_scores = []
    
    fold_metrics = {
        "verif_far": [], "verif_frr": [],
        "ident_fnir": [], "open_set_far": []
    }
    
    # Create figure: K rows, 4 COLUMNS
    # Col 1: Verif Dist, Col 2: Verif Err, Col 3: Ident Dist, Col 4: Ident Err
    fig, axes = plt.subplots(k_folds, 4, figsize=(28, 5 * k_folds))
    fig.suptitle(f'Cross-Validation Analysis ({k_folds}-Fold)', fontsize=20, y=0.99)
    plt.subplots_adjust(hspace=0.4, wspace=0.25)
    
    if k_folds == 1: axes = np.array([axes])

    for fold_idx, (train_index, val_index) in enumerate(skf.split(embeddings, labels)):
        if verbose: print(f"\n--- Fold {fold_idx + 1}/{k_folds} ---")
        
        # 1. Split Data & Compute Scores
        fold_gallery_emb, fold_gallery_lbl = embeddings[train_index], labels[train_index]
        fold_val_emb, fold_val_lbl = embeddings[val_index], labels[val_index]
        
        gen_scores, imp_scores_verif, imp_scores_ident = compute_scores_for_tuning(
            fold_val_emb, fold_val_lbl, fold_gallery_emb, fold_gallery_lbl
        )
        
        # 2. Calculate Curves
        thresholds = np.arange(0.0, 1.0, step)
        
        # A. Verification (1:1)
        frr_list = np.sum(gen_scores < thresholds[:, None], axis=1) / len(gen_scores)
        far_list_verif = np.sum(imp_scores_verif >= thresholds[:, None], axis=1) / len(imp_scores_verif)
        
        safe_indices_tv = np.where(far_list_verif <= desired_far)[0]
        selected_tv = thresholds[safe_indices_tv[0]] + bias_tv if len(safe_indices_tv) > 0 else 1.0
        
        # B. Identification (1:N)
        fnir_list = fnir_curve(fold_val_emb, fold_val_lbl, fold_gallery_emb, fold_gallery_lbl, thresholds)
        far_list_ident = np.sum(imp_scores_ident >= thresholds[:, None], axis=1) / len(imp_scores_ident)

        safe_indices_ti = np.where(far_list_ident <= desired_far)[0]
        selected_ti = thresholds[safe_indices_ti[0]] + bias_ti if len(safe_indices_ti) > 0 else 1.0

        tv_scores.append(selected_tv)
        ti_scores.append(selected_ti)

        # 3. Compute detailed metrics for printing/storage
        if verbose:
            print(f"   Tv: {selected_tv:.4f} | Ti: {selected_ti:.4f}")
            
        v_far, v_frr = compute_far_frr_verification_on_probes(selected_tv, fold_val_emb, fold_val_lbl, fold_gallery_emb, fold_gallery_lbl, verbose=verbose)
        i_fnir = compute_fnir_identification_on_probes(selected_ti, fold_val_emb, fold_val_lbl, fold_gallery_emb, fold_gallery_lbl, gal_names)
        o_far = compute_open_set_far_on_probes(threshold=selected_ti, probe_embs=fold_val_emb, probe_labels=fold_val_lbl, gal_embs=fold_gallery_emb, gal_labels=fold_gallery_lbl, gal_names=gal_names)
        
        fold_metrics["verif_far"].append(v_far)
        fold_metrics["verif_frr"].append(v_frr)
        fold_metrics["ident_fnir"].append(i_fnir)
        fold_metrics["open_set_far"].append(o_far)

        # 4. Call Visualization Helper
        scores_pack = {'gen': gen_scores, 'imp_verif': imp_scores_verif, 'imp_ident': imp_scores_ident}
        metrics_pack = {'far_verif': far_list_verif, 'frr': frr_list, 'far_ident': far_list_ident, 'fnir': fnir_list}
        
        plot_threshold_tuning_results(axes[fold_idx], fold_idx, scores_pack, metrics_pack, thresholds, selected_tv, selected_ti)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()
    
    # 5. Averages and Return
    avg_tv = np.mean(tv_scores) 
    avg_ti = np.mean(ti_scores) 
    std_tv = np.std(tv_scores) 
    std_ti = np.std(ti_scores)
    
    avg_metrics = {k: np.mean(v) for k, v in fold_metrics.items()}
    std_metrics = {k: np.std(v) for k, v in fold_metrics.items()}
    
    print("\n" + "="*60)
    print(f"   CROSS-VALIDATION SUMMARY (Average across {k_folds} folds)")
    print("="*60)
    print(f"{'Metric':<30} | {'Average':<10} | {'Std Dev':<10}")
    print("-" * 60)
    print(f"{'Verification FAR (Tv)':<30} | {avg_metrics['verif_far']*100:.2f}%     | ±{std_metrics['verif_far']*100:.2f}%")
    print(f"{'Verification FRR (Tv)':<30} | {avg_metrics['verif_frr']*100:.2f}%     | ±{std_metrics['verif_frr']*100:.2f}%")
    print("-" * 60)
    print(f"{'Identification Miss (FNIR)':<30} | {avg_metrics['ident_fnir']*100:.2f}%     | ±{std_metrics['ident_fnir']*100:.2f}%")
    print(f"{'Open-Set FAR (Ti)':<30} | {avg_metrics['open_set_far']*100:.2f}%     | ±{std_metrics['open_set_far']*100:.2f}%")
    print("="*60)
    print(f"Tv = {avg_tv:.4f} (std {std_tv:.4f}) | Ti = {avg_ti:.4f} (std {std_ti:.4f})")

    return {
        "Tv": avg_tv, "Ti": avg_ti, "std_Tv": std_tv, "std_Ti": std_ti,
        "fold_Tv": tv_scores, "fold_Ti": ti_scores, "metrics": avg_metrics,
    }








def plot_performance(probe_embs, probe_labels, gal_embs, gal_labels, Tv, Ti):
    """
    Generates SEPARATE Score Distributions and Error Rate Curves for the Test Set.
    Aligns with the 'Strict Security Logic' used in threshold tuning.
    
    Columns:
      1. Verification Distribution (Genuine vs All-Impostors)
      2. Verification Error Curves (FAR vs FRR)
      3. Identification Distribution (Genuine vs Max-Impostors)
      4. Identification Error Curves (Open-Set FAR vs FNIR)
    """
    print("--- Computing Score Distributions for Test Set ---")
    
    # 1. Compute Scores
    gen_scores, imp_scores_verif, imp_scores_ident = compute_scores_for_tuning(
        probe_embs, probe_labels, gal_embs, gal_labels
    )
    
    # 2. Calculate Metrics over Threshold Range
    thresholds = np.arange(0.0, 1.0, 0.001)
    
    # Verification Metrics (1:1)
    frr_list = np.sum(gen_scores < thresholds[:, None], axis=1) / len(gen_scores)
    far_list_verif = np.sum(imp_scores_verif >= thresholds[:, None], axis=1) / len(imp_scores_verif)
    
    # Identification Metrics (1:N): per-probe Rank-1 miss rate
    fnir_list = fnir_curve(probe_embs, probe_labels, gal_embs, gal_labels, thresholds)
    far_list_ident = np.sum(imp_scores_ident >= thresholds[:, None], axis=1) / len(imp_scores_ident)

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    fig.suptitle('Final Test Set Analysis (Probe vs Gallery) - Separate Security Logic', fontsize=18, y=0.98)
    plt.subplots_adjust(wspace=0.25)
    
    # --- Plot 1: Verification Distribution (Gen vs All-Imp) ---
    ax_v_dist = axes[0]
    sns.histplot(gen_scores, kde=True, bins=50, color='green', label='Genuine', 
                 stat='density', element="step", alpha=0.4, ax=ax_v_dist)
    sns.histplot(imp_scores_verif, kde=True, bins=50, color='red', label='Imp (All Pairs)', 
                 stat='density', element="step", alpha=0.4, ax=ax_v_dist)
    
    ax_v_dist.axvline(Tv, color='blue', linestyle='--', linewidth=2, label=f'Tv: {Tv:.4f}')
    ax_v_dist.set_title("Verification Distribution (1:1)", fontweight='bold')
    ax_v_dist.set_xlabel("Cosine Similarity")
    ax_v_dist.legend(loc='upper left', fontsize='small')
    ax_v_dist.grid(alpha=0.3)

    # --- Plot 2: Verification Curves (FAR vs FRR) ---
    ax_v_err = axes[1]
    ax_v_err.plot(thresholds, far_list_verif, color='red', linewidth=2, label='FAR (Verif)')
    ax_v_err.plot(thresholds, frr_list, color='green', linewidth=2, label='FRR')
    
    ax_v_err.axvline(Tv, color='blue', linestyle='--', label='Selected Tv')
    ax_v_err.set_title("Verification Error Rates", fontweight='bold')
    ax_v_err.set_xlabel("Threshold")
    ax_v_err.set_ylabel("Error Rate")
    ax_v_err.set_xlim(0, 1.0)
    ax_v_err.set_ylim(0, 0.2)
    ax_v_err.legend(loc='upper right', fontsize='small')
    ax_v_err.grid(True, linestyle='--', alpha=0.5)

    # --- Plot 3: Identification Distribution (Gen vs Max-Imp) ---
    ax_i_dist = axes[2]
    sns.histplot(gen_scores, kde=True, bins=50, color='green', label='Genuine', 
                 stat='density', element="step", alpha=0.4, ax=ax_i_dist)
    # The 'Max Impostor' distribution (Red) uses filled polygons for emphasis
    sns.histplot(imp_scores_ident, kde=True, bins=50, color='red', label='Imp (Max/Probe)', 
                 stat='density', element="step", fill=True, alpha=0.3, ax=ax_i_dist)
    
    ax_i_dist.axvline(Ti, color='orange', linestyle=':', linewidth=2, label=f'Ti: {Ti:.4f}')
    ax_i_dist.set_title("Identification Distribution (1:N)", fontweight='bold')
    ax_i_dist.set_xlabel("Cosine Similarity")
    ax_i_dist.legend(loc='upper left', fontsize='small')
    ax_i_dist.grid(alpha=0.3)

    # --- Plot 4: Identification Curves (Open-Set FAR vs FNIR) ---
    ax_i_err = axes[3]
    ax_i_err.plot(thresholds, far_list_ident, color='red', linewidth=2, label='Open-Set FAR')
    ax_i_err.plot(thresholds, fnir_list, color='green', linestyle='--', linewidth=2, label='FNIR')
    
    ax_i_err.axvline(Ti, color='orange', linestyle=':', label='Selected Ti')
    ax_i_err.set_title("Identification Error Rates", fontweight='bold')
    ax_i_err.set_xlabel("Threshold")
    ax_i_err.set_ylabel("Error Rate")
    ax_i_err.set_xlim(0, 1.0)
    ax_i_err.set_ylim(0, 0.2)
    ax_i_err.legend(loc='upper right', fontsize='small')
    ax_i_err.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    


    
    
def plot_threshold_tuning_results(axes, fold_idx, scores, metrics, thresholds, selected_tv, selected_ti):
    """
    Plots 4 charts for a specific fold:
      1. Verification Score Distribution (Genuine vs All-Impostors)
      2. Verification Error Curves (FAR vs FRR)
      3. Identification Score Distribution (Genuine vs Max-Impostors)
      4. Identification Error Curves (Open-Set FAR vs FNIR)
    
    Args:
        axes (list/array): List of 4 matplotlib axes for this row.
        fold_idx (int): Current fold number (0-based).
        scores (dict): Dictionary containing 'gen', 'imp_verif', 'imp_ident'.
        metrics (dict): Dictionary containing 'far_verif', 'frr', 'far_ident', 'fnir'.
        thresholds (np.array): X-axis values.
        selected_tv (float): Selected Verification Threshold.
        selected_ti (float): Selected Identification Threshold.
    """
    # Unpack Data
    gen_scores = scores['gen']
    imp_verif = scores['imp_verif']
    imp_ident = scores['imp_ident']
    
    far_verif = metrics['far_verif']
    frr = metrics['frr']
    far_ident = metrics['far_ident']
    fnir = metrics['fnir'] # Rank-1 miss rate (wrong identity or score below threshold)

    # --- Plot 1: Verification Distribution (Gen vs All-Imp) ---
    ax_v_dist = axes[0]
    sns.histplot(gen_scores, kde=True, bins=50, color='green', label='Genuine', 
                 stat='density', element="step", alpha=0.4, ax=ax_v_dist)
    sns.histplot(imp_verif, kde=True, bins=50, color='red', label='Imp (All Pairs)', 
                 stat='density', element="step", alpha=0.4, ax=ax_v_dist)
    
    ax_v_dist.axvline(selected_tv, color='blue', linestyle='--', linewidth=2, label=f'Tv: {selected_tv:.3f}')
    ax_v_dist.set_title(f"Fold {fold_idx+1}: Verif. Distribution", fontweight='bold')
    ax_v_dist.set_xlabel("Cosine Similarity")
    ax_v_dist.legend(loc='upper left', fontsize='small')
    ax_v_dist.grid(alpha=0.3)

    # --- Plot 2: Verification Curves (FAR vs FRR) ---
    ax_v_err = axes[1]
    ax_v_err.plot(thresholds, far_verif, color='red', linewidth=2, label='FAR (Verif)')
    ax_v_err.plot(thresholds, frr, color='green', linewidth=2, label='FRR')
    
    ax_v_err.axvline(selected_tv, color='blue', linestyle='--', label='Selected Tv')
    ax_v_err.set_title(f"Fold {fold_idx+1}: Verif. Error Rates", fontweight='bold')
    ax_v_err.set_xlabel("Threshold")
    ax_v_err.set_xlim(0, 1.0)
    ax_v_err.set_ylim(0, 0.2)
    ax_v_err.legend(loc='upper right', fontsize='small')
    ax_v_err.grid(True, linestyle='--', alpha=0.5)

    # --- Plot 3: Identification Distribution (Gen vs Max-Imp) ---
    # CRITICAL CHANGE: This plot now compares Genuine to the WORST CASE Impostor only
    ax_i_dist = axes[2]
    sns.histplot(gen_scores, kde=True, bins=50, color='green', label='Genuine', 
                 stat='density', element="step", alpha=0.4, ax=ax_i_dist)
    sns.histplot(imp_ident, kde=True, bins=50, color='red', label='Imp (Max/Probe)', 
                 stat='density', element="step", fill=True, alpha=0.3, ax=ax_i_dist)
    
    ax_i_dist.axvline(selected_ti, color='orange', linestyle=':', linewidth=2, label=f'Ti: {selected_ti:.3f}')
    ax_i_dist.set_title(f"Fold {fold_idx+1}: Ident. Distribution", fontweight='bold')
    ax_i_dist.set_xlabel("Cosine Similarity")
    ax_i_dist.legend(loc='upper left', fontsize='small')
    ax_i_dist.grid(alpha=0.3)

    # --- Plot 4: Identification Curves (Open-Set FAR vs FNIR) ---
    ax_i_err = axes[3]
    ax_i_err.plot(thresholds, far_ident, color='red', linewidth=2, label='Open-Set FAR')
    ax_i_err.plot(thresholds, fnir, color='green', linestyle='--', linewidth=2, label='FNIR')
    
    ax_i_err.axvline(selected_ti, color='orange', linestyle=':', label='Selected Ti')
    ax_i_err.set_title(f"Fold {fold_idx+1}: Ident. Error Rates", fontweight='bold')
    ax_i_err.set_xlabel("Threshold")
    ax_i_err.set_xlim(0, 1.0)
    ax_i_err.set_ylim(0, 0.2)
    ax_i_err.legend(loc='upper right', fontsize='small')
    ax_i_err.grid(True, linestyle='--', alpha=0.5)
"""
Utility functions: dataset loading, embedding extraction, gallery storage,
model-selection metrics and plotting helpers.
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from skimage.transform import resize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import Bunch, shuffle

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
FACENET_MODELS = {"InceptionResnetV1-VGGFace2": "vggface2", "InceptionResnetV1-CASIA": "casia-webface"}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def _list_images(folder):
    """Recursively lists the image files of a folder (sorted for reproducibility)."""
    paths = []
    for root, _, files in os.walk(folder):
        for file in sorted(files):
            if file.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(root, file))
    return sorted(paths)


def load_dataset(data_dir, split_report_name="dataset_split_report.csv"):
    """
    Scans the face-recognition dataset and returns FILE PATHS (images are loaded lazily).

    Expected structure:
        data_dir/galleries/subject_XX/**/*.jpg   -> enrolment (gallery / train) set
        data_dir/probes/subject_XX/**/*.jpg      -> probe (test) set

    Subjects whose folder name ends with 'X' are skipped. A CSV report with the
    gallery/probe assignment of every file is written to `data_dir`.

    Returns:
        tuple: (train_data, test_data, target_names), Bunches with `.images` (paths) and `.target`.
    """
    galleries_path = os.path.join(data_dir, "galleries")
    probes_path = os.path.join(data_dir, "probes")
    if not os.path.exists(galleries_path):
        raise FileNotFoundError(f"Could not find 'galleries' folder inside {data_dir}")

    train_paths, train_labels, test_paths, test_labels = [], [], [], []
    valid_subject_names, split_report = [], []

    gal_subjects = sorted(d for d in os.listdir(galleries_path) if os.path.isdir(os.path.join(galleries_path, d)))
    print(f"Found {len(gal_subjects)} subjects in 'galleries'. Scanning for paths...")

    label_id = 0
    for subject_name in gal_subjects:
        if subject_name.endswith('X'):
            print(f"Skipping subject: {subject_name}")
            continue

        gallery_files = _list_images(os.path.join(galleries_path, subject_name))
        if not gallery_files:
            print(f"Skipping {subject_name} (Gallery): No images found.")
            continue
        valid_subject_names.append(subject_name)

        for path in shuffle(gallery_files, random_state=42):
            train_paths.append(path); train_labels.append(label_id)
            split_report.append({"Subject": subject_name, "Label": label_id, "Path": path, "Split": "Gallery"})

        for path in _list_images(os.path.join(probes_path, subject_name)):
            test_paths.append(path); test_labels.append(label_id)
            split_report.append({"Subject": subject_name, "Label": label_id, "Path": path, "Split": "Probe"})

        label_id += 1

    pd.DataFrame(split_report).to_csv(os.path.join(data_dir, split_report_name), index=False)
    print(f"Split report saved to '{split_report_name}'")

    target_names = np.array(valid_subject_names)
    train_data = Bunch(images=np.array(train_paths), target=np.array(train_labels, dtype=np.int64), target_names=target_names)
    test_data = Bunch(images=np.array(test_paths), target=np.array(test_labels, dtype=np.int64), target_names=target_names)

    print(f"Dataset paths loaded. Gallery: {len(train_paths)} | Probes: {len(test_paths)}")
    return train_data, test_data, target_names


def load_demo_data(data_dir, max_samples=None, verbose=False):
    """
    Lists the images acquired for a live-demo session (one subject, unlabelled).

    Args:
        data_dir: Folder with the acquired samples.
        max_samples: Keep at most this many samples (sorted by file name).
        verbose: Print what was found.

    Returns:
        Bunch with `.images` (paths), dummy `.target` (-1) and `.target_names`.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory '{data_dir}' not found.")
    image_paths = _list_images(data_dir)
    if max_samples is not None and len(image_paths) > max_samples:
        if verbose: print(f"Found {len(image_paths)} samples, only the first {max_samples} are used.")
        image_paths = image_paths[:max_samples]
    if verbose: print(f"Using {len(image_paths)} sample(s) from '{data_dir}'.")
    return Bunch(images=np.array(image_paths, dtype=object),
                 target=np.full(len(image_paths), -1, dtype=int),
                 target_names=np.array(['Unknown']))


# ---------------------------------------------------------------------------
# Visualisation of raw data
# ---------------------------------------------------------------------------
def plot_gallery(images, titles, n_row=3, n_col=4):
    """Plots a grid of (already loaded) images."""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(min(len(images), n_row * n_col)):
        plt.subplot(n_row, n_col, i + 1)
        img = images[i]
        plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
        plt.title(titles[i], size=12)
        plt.xticks(()); plt.yticks(())
    plt.show()


def check_dataset(n_people_to_display, dataset, target_names):
    """Displays the images (loaded from their paths) of the first `n_people_to_display` subjects."""
    for label in np.unique(dataset.target)[:n_people_to_display]:
        person_paths = dataset.images[dataset.target == label]
        if target_names is not None and 0 <= label < len(target_names):
            person_name = str(target_names[label])
        else:
            person_name = f"ID_{label}"

        loaded, titles = [], []
        for i, path in enumerate(person_paths):
            try:
                with Image.open(path) as img:
                    loaded.append(np.array(img.convert('RGB')))
                    titles.append(f"{person_name} #{i+1}")
            except OSError:
                print(f"Warning: Could not load {path}")
        if loaded:
            plot_gallery(loaded, titles, n_row=1, n_col=len(loaded))


def visualize_detected_faces(processed_face_dataset, n_cols=10, verbose=False, images=False):
    """Shows every face of a processed Bunch (HWC or CHW, float in [0, 1] or uint8) in a grid."""
    n_kept = len(processed_face_dataset.images)
    if n_kept == 0:
        print("No faces were detected to display.")
        return

    n_rows = int(np.ceil(n_kept / n_cols))
    plt.figure(figsize=(2 * n_cols, 2.8 * n_rows))
    for k in range(n_kept):
        plt.subplot(n_rows, n_cols, k + 1)
        img_np = processed_face_dataset.images[k]
        if torch.is_tensor(img_np):
            img_np = img_np.detach().cpu().numpy()
        img_np = np.asarray(img_np)
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):      # CHW -> HWC
            img_np = np.transpose(img_np, (1, 2, 0))
        if img_np.ndim == 3 and img_np.shape[2] == 1:
            img_np = img_np.squeeze(axis=2)
        if np.issubdtype(img_np.dtype, np.floating):
            img_np = np.clip(img_np / 255.0 if img_np.max() > 1.0 else img_np, 0.0, 1.0)

        t_id = processed_face_dataset.target[k]
        names = processed_face_dataset.target_names
        label_name = str(names[t_id]) if names is not None and 0 <= t_id < len(names) else str(t_id)
        if verbose: print(f"Image {k}: Subject='{label_name}' (ID: {t_id})")

        plt.imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
        plt.title(f"{k}: {label_name}", fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.suptitle(f"{'Images' if images else 'Faces'} Detected ({n_kept} total)", fontsize=14, y=1.02)
    plt.show()


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def _to_hwc_unit(img):
    """Returns an HWC float32 image in [0, 1] from HWC/CHW, float/uint8 input."""
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def preprocess_and_embed(data, model_name, device, batch_size=32, verbose=False):
    """
    Extracts one embedding per face crop.

    Supports facenet-pytorch models ('InceptionResnetV1-VGGFace2', 'InceptionResnetV1-CASIA')
    and every DeepFace model name (ArcFace, Facenet512, VGG-Face, GhostFaceNet, SFace, ...).

    Args:
        data: Bunch whose `.images` are RGB face crops (HWC or CHW, values in [0, 1]).
        model_name: Name of the feature extractor.
        device: 'cpu' or 'cuda' (PyTorch models only).
        batch_size: Batch size (PyTorch models only).
        verbose: Print progress.

    Returns:
        np.ndarray of shape (N, D) with the raw (not normalised) embeddings.
    """
    images = data.images
    if verbose:
        print(f"--- Extracting features using [{model_name}] ---")

    # PATH A: facenet-pytorch InceptionResnetV1 (expects RGB, 160x160, values in [-1, 1])
    if model_name in FACENET_MODELS:
        from facenet_pytorch import InceptionResnetV1
        model = InceptionResnetV1(pretrained=FACENET_MODELS[model_name]).eval().to(device)

        embeddings = []
        for i in range(0, len(images), batch_size):
            batch = []
            for img in images[i:i + batch_size]:
                img = resize(_to_hwc_unit(img), (160, 160), anti_aliasing=True).astype(np.float32)
                img = img * 2.0 - 1.0
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                batch.append(np.transpose(img, (2, 0, 1)))       # HWC -> CHW
            with torch.no_grad():
                out = model(torch.tensor(np.array(batch), dtype=torch.float32, device=device))
            embeddings.append(out.cpu().numpy())

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.concatenate(embeddings, axis=0)

    # PATH B: DeepFace models
    from deepface import DeepFace  # heavy (TensorFlow) dependency, imported only when needed

    deepface_embeddings = []
    for i, img in enumerate(images):
        img_uint8 = (np.round(_to_hwc_unit(img) * 255)).astype(np.uint8)
        # DeepFace expects numpy inputs in OpenCV's BGR order, our crops are RGB.
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR) if img_uint8.ndim == 3 else img_uint8
        results = DeepFace.represent(img_path=img_bgr, model_name=model_name,
                                     enforce_detection=False, detector_backend="skip", align=False)
        deepface_embeddings.append(results[0]["embedding"])
        if verbose and i % 100 == 0:
            print(f"   Processed {i}/{len(images)}", end="\r")

    return np.array(deepface_embeddings)


# ---------------------------------------------------------------------------
# Model selection (on the gallery only, probes are kept as untouched test set)
# ---------------------------------------------------------------------------
def cross_validated_rank1(embeddings, labels, k_folds=4, seed=42):
    """
    Rank-1 identification accuracy estimated with Stratified K-Fold on the gallery:
    each fold is used as probes against the remaining K-1 folds (cosine similarity).

    Used for MODEL SELECTION so that the probe set is never looked at before the
    final evaluation.

    Args:
        embeddings: (N, D) L2-normalised gallery embeddings.
        labels: (N,) identity labels.

    Returns:
        tuple: (mean accuracy, std across folds)
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    accs = []
    for train_idx, val_idx in skf.split(embeddings, labels):
        sim = np.dot(embeddings[val_idx], embeddings[train_idx].T)
        pred = labels[train_idx][np.argmax(sim, axis=1)]
        accs.append(np.mean(pred == labels[val_idx]))
    return float(np.mean(accs)), float(np.std(accs))


def title(y_pred_test, y_test, target_names, i):
    pred_name = target_names[y_pred_test[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


def plot_losses(best_mlp):
    """Plots training loss and validation accuracy of an sklearn MLP trained with early_stopping=True."""
    if not hasattr(best_mlp, 'validation_scores_'):
        print("No validation scores available. Ensure 'early_stopping=True' is set.")
        return
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color='tab:blue')
    ax1.plot(best_mlp.loss_curve_, color='tab:blue', label='Training Loss')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy', color='tab:orange')
    ax2.plot(best_mlp.validation_scores_, color='tab:orange', linestyle='--', label='Validation Score')
    plt.title('Training Loss vs. Validation Accuracy')
    fig.tight_layout(); plt.grid(True); plt.show()


# ---------------------------------------------------------------------------
# Gallery (template database)
# ---------------------------------------------------------------------------
def create_gallery(features, labels, target_names, filename="gallery.npy", database_dir="./database/gallery", verbose=False):
    """
    Saves embeddings, labels and names into a single .npy file (a pickled dict).
    Only features are stored, never face images.
    """
    os.makedirs(database_dir, exist_ok=True)
    full_path = os.path.join(database_dir, filename)
    np.save(full_path, {"embeddings": features, "labels": labels, "names": target_names}, allow_pickle=True)
    if verbose:
        print(f"Gallery saved to '{full_path}': {features.shape} ({os.path.getsize(full_path) / 1024:.2f} KB)")


def load_gallery(filename="gallery.npy", gallery_path="./database/gallery"):
    """
    Loads the gallery file written by `create_gallery`.
    Only load galleries you created yourself (the file is a pickle).

    Returns:
        tuple: (embeddings, labels, names)
    """
    full_path = os.path.join(gallery_path, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Gallery file not found at: {full_path}")
    data = np.load(full_path, allow_pickle=True).item()
    return data['embeddings'], data['labels'], data['names']


def check_gallery_content(filename="gallery.npy", database_dir="./database/gallery", n_samples=20):
    """Prints shapes and a preview of the gallery content."""
    try:
        embeddings, labels, names = load_gallery(filename, database_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    print(f"\n--- Checking Gallery File: {filename} ---")
    print(f"Embeddings shape: {embeddings.shape} | Labels shape: {labels.shape} | Names shape: {names.shape}")
    for i in range(min(n_samples, len(labels))):
        lbl_id = labels[i]
        name_str = names[lbl_id] if lbl_id < len(names) else f"ID_{lbl_id}"
        print(f" Entry {i}: Label={lbl_id} ({name_str}) | Vector[:5]={embeddings[i][:5]}...")
    status = "PASSED" if len(embeddings) == len(labels) else f"FAILED ({len(embeddings)} vs {len(labels)})"
    print(f"\nIntegrity Check: {status}")


# ---------------------------------------------------------------------------
# Evaluation plots and metrics
# ---------------------------------------------------------------------------
def plot_normalized_confusion_matrix(y_true, y_pred, target_names, model_name):
    """Row-normalised confusion matrix (rows = true subject, columns = predicted subject)."""
    print(f"\n=== NORMALIZED CONFUSION MATRIX: {model_name} ===")
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Normalized Confusion Matrix - {model_name}\n(Values are % of True Class)", fontsize=14)
    plt.ylabel('True Identity (Ground Truth)', fontsize=12)
    plt.xlabel('Predicted Identity (System Output)', fontsize=12)
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout(); plt.show()


def plot_subject_average_similarity(sim_matrix, probe_labels, gallery_labels, target_names):
    """Average cosine similarity between every (probe subject, gallery subject) pair."""
    subjects = sorted(set(probe_labels) | set(gallery_labels))
    avg = np.zeros((len(subjects), len(subjects)))
    for i, s_row in enumerate(subjects):
        rows = np.where(probe_labels == s_row)[0]
        for j, s_col in enumerate(subjects):
            cols = np.where(gallery_labels == s_col)[0]
            if len(rows) and len(cols):
                avg[i, j] = np.mean(sim_matrix[np.ix_(rows, cols)])

    tick_labels = [target_names[i] for i in subjects] if target_names is not None else subjects
    plt.figure(figsize=(12, 10))
    sns.heatmap(avg, annot=True, fmt='.2f', cmap='viridis', vmin=-1, vmax=1,
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title("Average Cosine Similarity (Subject vs Subject)")
    plt.xlabel("Gallery Subject"); plt.ylabel("Probe Subject")
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout(); plt.show()


def evaluate_rank_n_performance(gallery_embs, gallery_labels, probe_embs, probe_labels, threshold, rank=1, gallery_names=None):
    """
    Identification performance at Rank-N for genuine (enrolled) probes.

      - Accuracy: the correct identity is in the Top-N AND its score >= threshold.
      - FRR: the correct identity is not returned (score < threshold or not in the Top-N
        while no wrong identity is accepted).
      - FAR: the correct identity is not in the Top-N but a WRONG identity has score >= threshold.

    Returns:
        tuple: (accuracy, frr, far)
    """
    sim_matrix = np.dot(probe_embs, gallery_embs.T)
    total = len(probe_embs)
    correct, false_rejections, false_acceptances = 0, 0, 0

    def name_of(lbl):
        return gallery_names[lbl] if gallery_names is not None and lbl < len(gallery_names) else f"ID_{lbl}"

    print(f"\n{'='*20} PERFORMANCE REPORT (Rank-{rank}) {'='*20}")
    for i in range(total):
        true_id = probe_labels[i]
        order = np.argsort(sim_matrix[i])[::-1]
        top_labels = gallery_labels[order[:rank]]
        top_scores = sim_matrix[i][order[:rank]]

        if true_id in top_labels:
            pos = np.where(top_labels == true_id)[0][0]
            if top_scores[pos] >= threshold:
                correct += 1
            else:
                false_rejections += 1
                print(f"[FALSE REJECT] Probe {i} ({name_of(true_id)}) found at Rank-{pos+1} "
                      f"but Score {top_scores[pos]:.4f} < Threshold {threshold}")
        elif np.max(top_scores) >= threshold:
            false_acceptances += 1
            print(f"[FALSE ACCEPT] Probe {i} ({name_of(true_id)}) mistaken for "
                  f"{name_of(gallery_labels[order[0]])} with Score {sim_matrix[i][order[0]]:.4f}")
        else:
            false_rejections += 1

    accuracy, frr, far = correct / total, false_rejections / total, false_acceptances / total
    print(f"{'-'*60}")
    print(f"Rank-{rank} Accuracy: {accuracy*100:.2f}%")
    print(f"False Rejection Rate (FRR): {frr*100:.2f}%")
    print(f"False Acceptance Rate (FAR): {far*100:.2f}%")
    print(f"{'='*60}\n")
    return accuracy, frr, far


def plot_evaluation_results(model_names, series, title="Feature extractor comparison", ylabel="Accuracy"):
    """
    Grouped bar chart comparing feature extractors.

    Args:
        model_names: list of model names (x axis).
        series: dict {legend label: list of accuracies aligned with model_names}.
    """
    x = np.arange(len(model_names))
    width = 0.8 / max(1, len(series))
    plt.figure(figsize=(18, 9))
    for k, (label, values) in enumerate(series.items()):
        plt.bar(x - 0.4 + width * (k + 0.5), values, width, label=label, edgecolor='black', alpha=0.9)

    all_scores = [v for values in series.values() for v in values]
    if all_scores:
        plt.ylim(bottom=max(0, min(all_scores) - 0.05), top=1.01)
    plt.xticks(x, model_names, fontsize=11)
    plt.xlabel('Model Architecture', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=15, fontweight='bold')
    plt.grid(axis='y', linestyle='-', alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout(); plt.show()

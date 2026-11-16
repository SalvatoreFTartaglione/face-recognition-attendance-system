"""
Liveness (replay-attack) detection module.

A MobileNetV2 backbone (ImageNet initialisation, fully fine-tuned) with a single
logit output classifies face crops as Live (1) or Fake (0). The module also
contains the dataset loading/splitting utilities and evaluation helpers.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import Bunch
from torch.utils.data import Dataset
from torchvision import models

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')


class LivenessDatasetWrapper(Dataset):
    """Exposes a Bunch of CHW face crops (+ labels) as a PyTorch Dataset."""

    def __init__(self, processed_data):
        self.source = processed_data

    def __len__(self):
        return len(self.source.images)

    def __getitem__(self, idx):
        img = torch.as_tensor(np.asarray(self.source.images[idx], dtype=np.float32))
        lbl = torch.tensor(self.source.target[idx], dtype=torch.long)
        return img, lbl


class LivenessMobileNet(nn.Module):
    """MobileNetV2 with its 1000-class head replaced by a single liveness logit."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)
        self.backbone.classifier[1] = nn.Linear(in_features=1280, out_features=1)

    def forward(self, x):
        return self.backbone(x)


def load_liveness_model(model_path, device):
    """Builds the network and loads a trained state_dict (safe, weights-only loading)."""
    model = LivenessMobileNet(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model.to(device).eval()


def download_liveness_dataset(data_dir, train_split=0.6, val_split=0.2, test_split=0.2, seed=42):
    """
    Loads the liveness dataset (Real vs Fake) as file paths and performs a
    SUBJECT-DISJOINT train/val/test split (GroupShuffleSplit on the subject id),
    so a subject's genuine images and their replayed copies never end up in
    different splits.

    Expected structure:
        data_dir/{galleries,probes}/{fake,real}/subject_XX/*.jpg

    Note: the ratios are applied to *subjects* (groups), so image proportions
    are only approximately equal to the requested ones.

    Returns:
        tuple: (train_data, val_data, test_data, target_names)
    """
    classes = {'fake': 0, 'real': 1}
    target_names = np.array(['Fake', 'Real'])
    all_paths, all_labels, all_subjects = [], [], []

    for source in ['galleries', 'probes']:
        source_path = os.path.join(data_dir, source)
        if not os.path.exists(source_path):
            continue
        for class_name, label_id in classes.items():
            class_path = os.path.join(source_path, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Class path '{class_path}' does not exist. Skipping.")
                continue
            for subject_folder in sorted(os.listdir(class_path)):
                subject_path = os.path.join(class_path, subject_folder)
                if not os.path.isdir(subject_path):
                    continue
                for file in sorted(os.listdir(subject_path)):
                    if file.lower().endswith(IMAGE_EXTENSIONS):
                        all_paths.append(os.path.join(subject_path, file))
                        all_labels.append(label_id)
                        all_subjects.append(subject_folder)

    all_paths, all_labels, all_subjects = map(np.array, (all_paths, all_labels, all_subjects))
    print(f"Found {len(all_paths)} images across {len(np.unique(all_subjects))} subjects.")

    # 1st split: train vs (val + test)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_split, random_state=seed)
    train_idx, temp_idx = next(gss1.split(all_paths, all_labels, groups=all_subjects))
    # 2nd split: val vs test
    relative_test_size = test_split / (val_split + test_split)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test_size, random_state=seed)
    val_sub, test_sub = next(gss2.split(all_paths[temp_idx], all_labels[temp_idx], groups=all_subjects[temp_idx]))
    val_idx, test_idx = temp_idx[val_sub], temp_idx[test_sub]

    def create_bunch(idxs):
        return Bunch(images=all_paths[idxs], target=all_labels[idxs],
                     target_names=target_names, subjects=np.unique(all_subjects[idxs]))

    train_data, val_data, test_data = create_bunch(train_idx), create_bunch(val_idx), create_bunch(test_idx)

    train_subs, val_subs, test_subs = set(train_data.subjects), set(val_data.subjects), set(test_data.subjects)
    if train_subs & val_subs or train_subs & test_subs or val_subs & test_subs:
        print("CRITICAL WARNING: DATA LEAKAGE DETECTED (Subjects overlap)!")
    else:
        print("Success: No subject overlap between splits.")

    return train_data, val_data, test_data, target_names


@torch.no_grad()
def predict_liveness(loader, model, device):
    """Returns the liveness probabilities (sigmoid of the logit) for every sample of a loader."""
    model.eval().to(device)
    probs = []
    for batch in loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        probs.extend(torch.sigmoid(model(images.to(device))).cpu().view(-1).numpy())
    return np.array(probs)


def evaluate_liveness(loader, model, device, threshold=0.5, live_demo=False):
    """
    Evaluates the liveness model on a labelled DataLoader.

    Args:
        loader: DataLoader yielding (images, labels), 1 = live, 0 = fake.
        model: Trained liveness model.
        device: 'cuda' or 'cpu'.
        threshold: Score >= threshold -> Live.
        live_demo: If True, skips the plots.

    Returns:
        tuple: (all_probs, all_labels, flr, ffr) where FLR = False Live Rate
        (fakes accepted) and FFR = False Fake Rate (lives rejected).
    """
    # Single pass over the loader so scores and labels stay aligned even if it shuffles
    model.eval().to(device)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            all_probs.extend(torch.sigmoid(model(images.to(device))).cpu().view(-1).numpy())
            all_labels.extend(labels.numpy())
    all_probs, all_labels = np.array(all_probs), np.array(all_labels)
    predictions = (all_probs >= threshold).astype(int)

    acc = accuracy_score(all_labels, predictions)
    fake_mask, real_mask = all_labels == 0, all_labels == 1
    flr = np.mean(predictions[fake_mask] == 1) if fake_mask.any() else 0.0
    ffr = np.mean(predictions[real_mask] == 0) if real_mask.any() else 0.0
    acer = (flr + ffr) / 2

    if not live_demo:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # A. Score distribution
        ax = axes[0]
        if real_mask.any():
            ax.hist(all_probs[real_mask], bins=30, alpha=0.5, color='green', label='Live', density=True)
        if fake_mask.any():
            ax.hist(all_probs[fake_mask], bins=30, alpha=0.5, color='red', label='Fake', density=True)
        ax.axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Tf={threshold:.3f}')
        ax.set_title("Score Distribution", fontweight='bold')
        ax.set_xlabel("Liveness Probability")
        ax.legend(); ax.grid(alpha=0.3)

        # B. ROC curve
        if len(np.unique(all_labels)) > 1:
            fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs)
            ax = axes[1]
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc(fpr, tpr):.3f}')
            ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            idx = np.argmin(np.abs(roc_thresholds - threshold))
            ax.scatter(fpr[idx], tpr[idx], color='black', zorder=5, label='Current Tf')
            ax.set_title("ROC Curve", fontweight='bold')
            ax.set_xlabel('FLR (False Live Rate)'); ax.set_ylabel('True Live Rate')
            ax.legend(); ax.grid(alpha=0.3)

        # C. Error rates vs threshold
        ts = np.linspace(0, 1, 200)
        fakes, lives = all_probs[fake_mask], all_probs[real_mask]
        flr_curve = [np.mean(fakes >= t) if len(fakes) else 0.0 for t in ts]
        ffr_curve = [np.mean(lives < t) if len(lives) else 0.0 for t in ts]
        ax = axes[2]
        ax.plot(ts, flr_curve, color='red', lw=2, label='FLR (False Live)')
        ax.plot(ts, ffr_curve, color='blue', lw=2, label='FFR (False Fake)')
        ax.axvline(threshold, color='black', linestyle='--', label=f'Current Tf={threshold:.3f}')
        ax.set_title("Error Rates vs Threshold", fontweight='bold')
        ax.set_xlabel('Threshold'); ax.set_ylabel('Error Rate')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(); ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

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
    Loads an external liveness test set made of `data_dir/fake/*` (label 0) and
    `data_dir/real/*` (label 1). Returns a Bunch of file paths and labels.
    """
    classes = {'fake': 0, 'real': 1}
    target_names = np.array(['Fake', 'Real'])
    all_paths, all_labels = [], []

    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return Bunch(images=np.array([]), target=np.array([]), target_names=target_names)

    for class_name, label_id in classes.items():
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Folder '{class_name}' not found in {data_dir}")
            continue
        for root, _, files in os.walk(class_path):
            for file in sorted(files):
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    all_paths.append(os.path.join(root, file))
                    all_labels.append(label_id)

    all_paths, all_labels = np.array(all_paths), np.array(all_labels)
    print(f"Found {len(all_paths)} images (Real: {np.sum(all_labels == 1)}, Fake: {np.sum(all_labels == 0)}).")
    return Bunch(images=all_paths, target=all_labels, target_names=target_names)


def filter_non_live_faces(processed_data, loader, model, device, threshold=0.5, verbose=False):
    """
    Scores every detected face and keeps only the ones classified as Live.

    Args:
        processed_data: Bunch returned by `face_detection.detect_faces(..., cnn=True)`
            (CHW faces, aligned `.paths` and `.target`).
        loader: DataLoader over `processed_data` with shuffle=False.
        model: Trained liveness model.
        device: 'cuda' or 'cpu'.
        threshold: Liveness threshold Tf (score >= Tf -> Live).
        verbose: Print the score and decision of every sample.

    Returns:
        Bunch with the live faces in HWC format (ready for display/embedding),
        their `.paths`, `.target` and `.liveness_scores`. It may be empty.
    """
    probs = predict_liveness(loader, model, device)
    if len(probs) != len(processed_data.images):
        raise ValueError(f"Mismatch: {len(probs)} scores vs {len(processed_data.images)} faces.")

    live_mask = probs >= threshold
    if verbose:
        print(f"\nLiveness check (Tf={threshold}):")
        for path, p in zip(processed_data.paths, probs):
            print(f"  {os.path.basename(str(path))}: score={p:.4f} -> {'LIVE' if p >= threshold else 'FAKE'}")
        print(f"  Live: {live_mask.sum()} | Fake (removed): {(~live_mask).sum()}")

    # CHW -> HWC copies (the input Bunch is left untouched)
    live_faces = np.empty(int(live_mask.sum()), dtype=object)
    for k, idx in enumerate(np.where(live_mask)[0]):
        live_faces[k] = np.transpose(processed_data.images[idx], (1, 2, 0))

    return Bunch(
        images=live_faces,
        target=processed_data.target[live_mask],
        paths=processed_data.paths[live_mask],
        liveness_scores=probs[live_mask],
        target_names=processed_data.target_names,
    )

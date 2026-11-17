"""
Face Image Quality Assessment (FIQA) with MagFace.

MagFace learns embeddings whose L2 magnitude is proportional to the sample's
recognisability, so the norm of the (unnormalised) MagFace feature is used as a
quality score. The quality threshold is calibrated with an Error-vs-Reject
Characteristic (ERC) analysis, using the system's own recognition model
(InceptionResnetV1-VGGFace2) to measure the error.
"""
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from facenet_pytorch import MTCNN
from skimage import transform as trans
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.utils import Bunch
from torchvision import transforms

from iresnet import iresnet50

# ArcFace/InsightFace reference positions of the 5 landmarks in a 112x112 crop
REFERENCE_LANDMARKS_112 = np.array([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.6963],  # Right eye
    [56.0252, 71.7366],  # Nose
    [41.5493, 92.3655],  # Left mouth corner
    [70.7299, 92.3655],  # Right mouth corner
], dtype=np.float32)


def _load_checkpoint(model_path, device):
    """Loads a checkpoint with weights-only unpickling, falling back to full unpickling (trusted files only)."""
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        warnings.warn(f"'{model_path}' is not a weights-only checkpoint: loading it with full unpickling. "
                      "Only do this with the official MagFace weights.")
        return torch.load(model_path, map_location=device, weights_only=False)


class MagFaceEvaluator:
    """Wraps a MagFace iResNet50 backbone and exposes a `quality()` score."""

    def __init__(self, model_path, architecture='iresnet50', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if architecture != 'iresnet50':
            raise ValueError(f"Unsupported architecture '{architecture}': only 'iresnet50' is available.")
        self.model = iresnet50(pretrained=False, num_classes=512)

        print(f"Loading MagFace weights from {model_path}...")
        checkpoint = _load_checkpoint(model_path, self.device)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))

        # The official checkpoint wraps the backbone (e.g. 'module.features.conv1.weight'):
        # detect the prefix from a known layer and strip it.
        prefix = next((k[:-len("conv1.weight")] for k in state_dict if k.endswith("conv1.weight")), None)
        if prefix is None:
            raise RuntimeError(f"Could not find 'conv1.weight' in the checkpoint. First keys: {list(state_dict)[:5]}")
        state_dict = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}

        # strict=False: the classification head of the training checkpoint is not needed
        missing, _ = self.model.load_state_dict(state_dict, strict=False)
        if 'conv1.weight' in missing:
            raise RuntimeError("MagFace backbone weights were not loaded correctly.")
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @torch.no_grad()
    def quality(self, aligned_face_rgb):
        """Quality score (MagFace feature magnitude) of an aligned 112x112 RGB uint8 face."""
        input_tensor = self.transform(aligned_face_rgb).unsqueeze(0).to(self.device)
        return torch.norm(self.model(input_tensor), p=2, dim=1).item()


def align_face_magface_mtcnn(img_rgb, detector):
    """
    Aligns the main face of an RGB image to the 112x112 MagFace template using
    the 5 MTCNN landmarks and a similarity transform.

    Returns:
        tuple: (aligned 112x112 image or None, success flag)
    """
    _, _, landmarks = detector.detect(img_rgb, landmarks=True)
    if landmarks is None or len(landmarks) == 0:
        return None, False
    tform = trans.SimilarityTransform()
    tform.estimate(landmarks[0].astype(np.float32), REFERENCE_LANDMARKS_112)
    aligned_img = cv2.warpAffine(img_rgb, tform.params[0:2, :], (112, 112), borderValue=0.0)
    return aligned_img, True


def compute_quality_scores(magface, image_paths, device, aligner=None, verbose=False):
    """
    MagFace quality of every image file. Images that cannot be read or aligned get NaN.
    """
    if aligner is None:
        aligner = MTCNN(keep_all=False, device=device)
    scores = np.full(len(image_paths), np.nan)
    for i, path in enumerate(image_paths):
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            if verbose: print(f"[Quality] Load failed: {path}")
            continue
        aligned, ok = align_face_magface_mtcnn(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), aligner)
        if not ok:
            if verbose: print(f"[Quality] Alignment failed: {path}")
            continue
        scores[i] = magface.quality(aligned)
    return scores


def select_samples_by_quality(magface, processed_dataset, quality_threshold, device, verbose=False):
    """
    Acquisition-module quality gate.

    Every acquired sample is scored with MagFace; samples whose score does not
    exceed `quality_threshold` (or that cannot be aligned) are DISCARDED. The
    accepted samples are returned sorted by decreasing quality, so the caller can
    use the best one (matching) or all of them (enrolment). If the returned Bunch
    is empty the user must be asked to re-acquire the sample(s).

    Args:
        magface: MagFaceEvaluator instance.
        processed_dataset: Bunch of detected faces with aligned `.paths` (from detect_faces /
            filter_non_live_faces).
        quality_threshold: Minimum MagFace score.
        device: Torch device for the aligner.
        verbose: Print the score/decision of every sample.

    Returns:
        Bunch with `.images`, `.target`, `.paths`, `.quality_scores` of the accepted samples
        (best first) and `.all_quality_scores` of every input sample.
    """
    scores = compute_quality_scores(magface, processed_dataset.paths, device, verbose=verbose)
    accepted = np.where(np.nan_to_num(scores, nan=-np.inf) > quality_threshold)[0]
    accepted = accepted[np.argsort(-scores[accepted])]

    if verbose:
        print(f"\nQuality check (threshold={quality_threshold:.3f}):")
        for path, s in zip(processed_dataset.paths, scores):
            status = "N/A (no aligned face)" if np.isnan(s) else ("PASS" if s > quality_threshold else "FAIL")
            print(f"  {path}: quality={s:.3f} -> {status}")

    images = np.empty(len(accepted), dtype=object)
    for k, idx in enumerate(accepted):
        images[k] = processed_dataset.images[idx]
    return Bunch(images=images,
                 target=np.asarray(processed_dataset.target)[accepted],
                 paths=np.asarray(processed_dataset.paths)[accepted],
                 quality_scores=scores[accepted],
                 all_quality_scores=scores,
                 target_names=processed_dataset.target_names)


def compute_fnmr_at_fmr(embeddings_probe, targets_probe, embeddings_gallery=None, targets_gallery=None, target_fmr=0.001):
    """
    False Non-Match Rate at a fixed False Match Rate (L2-normalised embeddings).

      1. Probe vs Gallery (bipartite) when gallery embeddings/targets are given.
      2. All-vs-all within the probe set otherwise (each unordered pair counted once).
    """
    embeddings_probe = np.asarray(embeddings_probe)
    if embeddings_probe.ndim == 3:
        embeddings_probe = embeddings_probe.squeeze(1)
    targets_probe = np.asarray(targets_probe)

    if embeddings_gallery is not None and targets_gallery is not None:
        embeddings_gallery = np.asarray(embeddings_gallery)
        if embeddings_gallery.ndim == 3:
            embeddings_gallery = embeddings_gallery.squeeze(1)
        sim_matrix = np.dot(embeddings_probe, embeddings_gallery.T)
        match_matrix = targets_probe[:, None] == np.asarray(targets_gallery)[None, :]
        genuine_scores, impostor_scores = sim_matrix[match_matrix], sim_matrix[~match_matrix]
    else:
        sim_matrix = np.dot(embeddings_probe, embeddings_probe.T)
        upper = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)     # exclude self/duplicate pairs
        match_matrix = targets_probe[:, None] == targets_probe[None, :]
        genuine_scores, impostor_scores = sim_matrix[upper & match_matrix], sim_matrix[upper & ~match_matrix]

    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return 1.0

    # Threshold that lets through `target_fmr` of the impostor pairs
    impostor_scores = np.sort(impostor_scores)[::-1]
    fmr_index = int(target_fmr * len(impostor_scores))
    threshold = impostor_scores[fmr_index] if fmr_index < len(impostor_scores) else 1.0
    return float(np.mean(genuine_scores < threshold))


def merge_quality_and_embeddings(magface, mtcnn_aligner_magface, device, dataset, precomputed_embs, desc, verbose=True):
    """
    Pairs the MagFace quality of every image file with its pre-computed recognition embedding.
    Images that MagFace cannot align are dropped.

    Args:
        dataset: Bunch with `.images` (file paths, aligned with `precomputed_embs`) and `.target`.
        precomputed_embs: (N, D) embeddings of the recognition model.

    Returns:
        tuple: (embeddings (N_valid, D), qualities (N_valid,), targets (N_valid,))
    """
    if len(dataset.images) != len(precomputed_embs):
        raise ValueError(f"Mismatch in {desc}: {len(dataset.images)} images vs {len(precomputed_embs)} embeddings.")

    qualities = compute_quality_scores(magface, dataset.images, device, aligner=mtcnn_aligner_magface, verbose=verbose)
    valid = ~np.isnan(qualities)
    print(f"[{desc}] Valid (image, embedding) pairs: {valid.sum()}/{len(valid)}")
    return np.asarray(precomputed_embs)[valid], qualities[valid], np.asarray(dataset.target)[valid]


def erc_curve(probe_embs, probe_qualities, probe_targets, gal_embs, gal_targets, reject_ratios, target_fmr=0.001):
    """
    Error-vs-Reject Characteristic: for each reject ratio R, the R% lowest-quality
    probes are discarded and the FNMR@FMR of the surviving probes is computed.

    Returns:
        tuple: (fnmr per ratio, quality threshold per ratio)
    """
    order = np.argsort(probe_qualities)
    n = len(probe_qualities)
    fnmr, thresholds = [], []
    for ratio in reject_ratios:
        n_drop = int(n * ratio)
        keep = order[n_drop:]
        thresholds.append(probe_qualities[order[n_drop - 1]] if n_drop > 0 else np.min(probe_qualities))
        fnmr.append(1.0 if len(keep) == 0 else compute_fnmr_at_fmr(
            probe_embs[keep], probe_targets[keep], gal_embs, gal_targets, target_fmr))
    return np.array(fnmr), np.array(thresholds)


def select_reject_ratio(reject_ratios, error_rates, max_reject=0.30, refine_tolerance=0.05):
    """
    Anchor-and-refine selection of the operating reject ratio.

      1. Anchor: the first ratio (below `max_reject`) whose error is within
         max(1%, 1.2 x global minimum); if none, the ratio with the lowest error
         below `max_reject`.
      2. Refine: among the ratios between the anchor and `max_reject`, keep those
         whose error is <= anchor_error + refine_tolerance and pick the largest
         (i.e. the strictest quality gate).

    NOTE: `refine_tolerance` is an ABSOLUTE difference of error rate:
    0.05 means +5 percentage points of FNMR (not +0.05%). With such a loose
    tolerance almost every ratio up to `max_reject` qualifies, so the selected
    ratio tends to sit right below `max_reject`; use a smaller value (e.g. 0.0005)
    to really keep the error "equal to the anchor".

    Returns:
        tuple: (selected index, anchor index, base target error)
    """
    error_rates = np.asarray(error_rates, dtype=float)
    target_performance = max(0.01, float(np.min(error_rates)) * 1.20)
    hard_limit_idx = int(np.searchsorted(reject_ratios, max_reject))

    below = np.where(error_rates[:hard_limit_idx] <= target_performance)[0]
    anchor_idx = int(below[0]) if len(below) else int(np.argmin(error_rates[:hard_limit_idx]))

    local_tolerance = error_rates[anchor_idx] + refine_tolerance
    candidates = [i for i in range(anchor_idx, hard_limit_idx) if error_rates[i] <= local_tolerance]
    return int(max(candidates)), anchor_idx, target_performance


def _plot_erc(reject_ratios, error_rates, sel_idx, anchor_idx, target_perf, threshold, title, error_std=None):
    plt.figure(figsize=(10, 6))
    plt.plot(reject_ratios * 100, error_rates, marker='o', linewidth=2, color='blue', label='FNMR Curve')
    if error_std is not None:
        plt.fill_between(reject_ratios * 100, error_rates - error_std, error_rates + error_std,
                         color='blue', alpha=0.15, label='±1 std (folds)')
    plt.axhline(y=np.min(error_rates), color='green', linestyle=':', label=f'Global Min ({np.min(error_rates):.2%})')
    plt.axhline(y=target_perf, color='orange', linestyle='--', label=f'Base Target ({target_perf:.2%})')
    plt.axvline(x=reject_ratios[sel_idx] * 100, color='red', linestyle='--', label=f"Cutoff ({reject_ratios[sel_idx]*100:.0f}%)")
    plt.scatter([reject_ratios[anchor_idx] * 100], [error_rates[anchor_idx]], color='gray', s=50, label='Anchor', zorder=5)
    plt.scatter([reject_ratios[sel_idx] * 100], [error_rates[sel_idx]], color='red', s=150, label='Selected', zorder=6)
    plt.text(reject_ratios[sel_idx] * 100 + 1, error_rates[sel_idx], f" Thresh: {threshold:.3f}", color='red', fontweight='bold')
    plt.title(title)
    plt.xlabel("Reject Ratio (%)"); plt.ylabel("FNMR")
    plt.grid(True, linestyle='--', alpha=0.6); plt.legend(loc='best')
    plt.tight_layout(); plt.show()


def compute_magface_threshold(magface, gallery_data, gallery_embeddings, device, k_folds=4, target_fmr=0.001,
                              max_reject=0.30, refine_tolerance=0.05, seed=42, verbose=True):
    """
    Calibrates the quality threshold on the GALLERY ONLY (the probe set stays an
    untouched test set) with Stratified K-Fold ERC analysis:

      - in every fold, the held-out gallery images act as probes and are matched
        against the remaining gallery images (recognition embeddings);
      - the ERC curves are averaged across folds and the reject ratio is chosen
        with `select_reject_ratio`;
      - the threshold is the MagFace quality at that ratio of the gallery quality distribution.

    Args:
        gallery_data: Bunch with `.images` (file paths aligned with `gallery_embeddings`) and `.target`.
        gallery_embeddings: (N, D) embeddings of the recognition model.

    Returns:
        float: quality threshold (samples must score ABOVE it).
    """
    aligner = MTCNN(keep_all=False, device=device)
    embs, qualities, targets = merge_quality_and_embeddings(magface, aligner, device, gallery_data,
                                                            gallery_embeddings, "Gallery", verbose=False)
    embs = normalize(embs, norm='l2')

    reject_ratios = np.round(np.arange(0, 0.51, 0.01), 2)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_curves = []
    for gal_idx, val_idx in skf.split(embs, targets):
        fnmr, _ = erc_curve(embs[val_idx], qualities[val_idx], targets[val_idx],
                            embs[gal_idx], targets[gal_idx], reject_ratios, target_fmr)
        fold_curves.append(fnmr)
    fold_curves = np.array(fold_curves)
    mean_curve, std_curve = fold_curves.mean(axis=0), fold_curves.std(axis=0)

    sel_idx, anchor_idx, target_perf = select_reject_ratio(reject_ratios, mean_curve, max_reject, refine_tolerance)
    sorted_q = np.sort(qualities)
    n_drop = int(len(sorted_q) * reject_ratios[sel_idx])
    threshold = float(sorted_q[n_drop - 1]) if n_drop > 0 else float(sorted_q[0])

    if verbose:
        print(f"{'Reject Ratio':<14} | {'Mean FNMR':<10} | {'Std':<8}")
        for r, m, s in zip(reject_ratios, mean_curve, std_curve):
            print(f"{r*100:>6.0f}%        | {m:>8.3%}  | {s:.4f}")
    _plot_erc(reject_ratios, mean_curve, sel_idx, anchor_idx, target_perf, threshold,
              f"ERC on gallery ({k_folds}-fold CV)\nSelected: {mean_curve[sel_idx]:.2%} @ quality {threshold:.3f} "
              f"(rejects {reject_ratios[sel_idx]:.0%} of the gallery samples)", error_std=std_curve)
    print(f"\nOptimal Quality Threshold: {threshold:.4f} | Reject ratio (CV): {reject_ratios[sel_idx]*100:.0f}% "
          f"| Expected FNMR@FMR={target_fmr:.1%}: {mean_curve[sel_idx]:.4%}")
    return threshold


def evaluate_quality_threshold_on_probes(magface, gallery_data, gallery_embeddings, probe_data, probe_embeddings,
                                         device, quality_threshold, target_fmr=0.001):
    """
    TEST-SET assessment of a quality threshold calibrated on the gallery: plots the
    ERC of the probes and reports the rejection rate and FNMR@FMR obtained with
    and without the quality gate.

    Returns:
        dict with 'reject_rate', 'fnmr_without_gate', 'fnmr_with_gate'.
    """
    aligner = MTCNN(keep_all=False, device=device)
    gal_embs, _, gal_t = merge_quality_and_embeddings(magface, aligner, device, gallery_data, gallery_embeddings, "Gallery", verbose=False)
    prb_embs, prb_q, prb_t = merge_quality_and_embeddings(magface, aligner, device, probe_data, probe_embeddings, "Probe", verbose=False)
    gal_embs, prb_embs = normalize(gal_embs, norm='l2'), normalize(prb_embs, norm='l2')

    keep = prb_q > quality_threshold
    result = {
        "reject_rate": float(1 - keep.mean()),
        "fnmr_without_gate": compute_fnmr_at_fmr(prb_embs, prb_t, gal_embs, gal_t, target_fmr),
        "fnmr_with_gate": compute_fnmr_at_fmr(prb_embs[keep], prb_t[keep], gal_embs, gal_t, target_fmr),
    }

    reject_ratios = np.round(np.arange(0, 0.51, 0.01), 2)
    fnmr, _ = erc_curve(prb_embs, prb_q, prb_t, gal_embs, gal_t, reject_ratios, target_fmr)
    plt.figure(figsize=(10, 6))
    plt.plot(reject_ratios * 100, fnmr, marker='o', linewidth=2, color='blue', label='Probe ERC')
    plt.axvline(result["reject_rate"] * 100, color='red', linestyle='--',
                label=f"Quality gate {quality_threshold:.3f} -> rejects {result['reject_rate']:.1%}")
    plt.title("ERC on probes (test set)")
    plt.xlabel("Reject Ratio (%)"); plt.ylabel("FNMR")
    plt.grid(True, linestyle='--', alpha=0.6); plt.legend(); plt.tight_layout(); plt.show()

    print(f"Probe reject rate: {result['reject_rate']:.2%}")
    print(f"FNMR@FMR={target_fmr:.1%} without gate: {result['fnmr_without_gate']:.4%} | "
          f"with gate: {result['fnmr_with_gate']:.4%}")
    return result

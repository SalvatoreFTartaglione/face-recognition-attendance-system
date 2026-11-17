"""
End-to-end attendance pipeline used by the live demo (notebook or command line).

    acquisition (<= 3 samples) -> face detection (MTCNN) -> [liveness check]
    -> quality gate (MagFace) -> embedding (InceptionResnetV1-VGGFace2)
    -> Enrollment | Identification (1:N, open set) | Verification (1:1)

Command-line usage:
    python live_system_demo.py identify --data-dir Test/new_user
    python live_system_demo.py verify   --data-dir Test/new_user --claimed-id 3
    python live_system_demo.py enroll   --data-dir Test/new_user --name new_user --no-liveness
"""
import argparse
import os
import time

import numpy as np
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

import face_detection as detector
import face_quality
import liveness_detector
import threshold_tuning as th_tuning
import utilities as utils

# Operating points calibrated in 01_face_authentication.ipynb (see README, "Results").
DEFAULT_CONFIG = {
    "Tf": 0.5000,                      # liveness threshold
    "Ti": 0.6833,                      # identification threshold
    "Tv": 0.6013,                      # verification threshold
    "quality_threshold": 20.264,       # MagFace quality threshold
    "max_samples": 3,                  # at most 3 acquisitions per session (system requirement)
    "detector": "MTCNN",
    "embedding_model": "InceptionResnetV1-VGGFace2",
    "batch_size": 32,
    "model_dir": "models",
    "liveness_model_name": "liveness_model.pth",
    "magface_model_name": "magface_iresnet50_MS1MV2_ddp_fp32.pth",
    "database_dir": "System_Database/gallery",
    "gallery_name": "gallery.npy",
}


class AcquisitionError(Exception):
    """Raised when no usable sample is available: the user must re-acquire."""


def _show_faces(faces, title):
    import matplotlib.pyplot as plt
    n = len(faces.images)
    plt.figure(figsize=(3 * n, 3.4))
    for k in range(n):
        plt.subplot(1, n, k + 1)
        plt.imshow(np.clip(np.asarray(faces.images[k], dtype=np.float32), 0, 1))
        plt.title(os.path.basename(str(faces.paths[k])), fontsize=8)
        plt.axis('off')
    plt.suptitle(title); plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def acquire_and_detect(data_dir, detector_method, max_samples, cnn):
    """Loads at most `max_samples` acquisitions and detects the face in each of them."""
    samples = utils.load_demo_data(data_dir, max_samples=max_samples, verbose=True)
    if len(samples.images) == 0:
        raise AcquisitionError(f"No images found in '{data_dir}'.")
    faces = detector.detect_faces(samples, method=detector_method, verbose=True, cnn=cnn)
    if len(faces.images) == 0:
        raise AcquisitionError("No face detected in the acquired samples.")
    return faces


def verify_liveness(faces_chw, liveness_model, device, batch_size, Tf):
    """
    Replay-attack check on CHW face crops. Fake samples are discarded.
    Returns the live faces in HWC format with their liveness scores.
    """
    loader = DataLoader(liveness_detector.LivenessDatasetWrapper(faces_chw), batch_size=batch_size, shuffle=False)
    live = liveness_detector.filter_non_live_faces(faces_chw, loader, liveness_model, device, Tf, verbose=True)
    if len(live.images) == 0:
        raise AcquisitionError("All samples were classified as FAKE (presentation attack suspected).")
    return live


def quality_gate(faces, magface, quality_threshold, device):
    """Keeps only the samples above the MagFace quality threshold (best first)."""
    accepted = face_quality.select_samples_by_quality(magface, faces, quality_threshold, device, verbose=True)
    if len(accepted.images) == 0:
        best = np.nanmax(accepted.all_quality_scores) if np.any(~np.isnan(accepted.all_quality_scores)) else float('nan')
        raise AcquisitionError(f"No sample reached the quality threshold ({quality_threshold:.3f}, best score: "
                               f"{best:.3f}). Please re-acquire in better conditions (lighting, pose, focus).")
    return accepted


def extract_embeddings(faces, model_name, device, batch_size):
    """L2-normalised embeddings of the given faces, shape (n, 512)."""
    return normalize(utils.preprocess_and_embed(faces, model_name, device, batch_size), norm='l2')


# ---------------------------------------------------------------------------
# Operating modes
# ---------------------------------------------------------------------------
def enroll_new_user(new_embeddings, new_name, gallery_path="gallery.npy", database_dir="./database/gallery"):
    """
    Appends a new user to the gallery, storing the features (never the images)
    of ALL the accepted samples. Creates the gallery if it does not exist.

    Returns:
        int: the ID assigned to the new user.
    """
    new_embeddings = np.atleast_2d(new_embeddings)
    full_path = os.path.join(database_dir, gallery_path)

    if os.path.exists(full_path):
        current_embs, current_labels, current_names = utils.load_gallery(gallery_path, database_dir)
        if new_name in current_names:
            existing = int(np.where(current_names == new_name)[0][0])
            raise ValueError(f"User '{new_name}' already exists (ID: {existing}). Enrollment skipped.")
        new_label = len(current_names)
        embs = np.vstack((current_embs, new_embeddings))
        labels = np.append(current_labels, [new_label] * len(new_embeddings))
        names = np.append(current_names, new_name)
    else:
        new_label = 0
        embs, labels, names = new_embeddings, np.zeros(len(new_embeddings), dtype=int), np.array([new_name])

    utils.create_gallery(embs, labels, names, filename=gallery_path, database_dir=database_dir)
    print(f"Enrollment completed: '{new_name}' -> ID {new_label} ({len(new_embeddings)} template(s) stored).")
    return new_label


def verify_user(embedding_norm, Tv, claimed_id, gallery_name, database_dir):
    """
    1:1 verification against the templates of the claimed identity only.

    Returns:
        tuple: (score, decision) with decision in {"MATCH", "REJECT"}.
    """
    gal_embs, gal_labels, gal_names = utils.load_gallery(filename=gallery_name, gallery_path=database_dir)
    score, decision = th_tuning.verify_user(claimed_id, embedding_norm, gal_embs, gal_labels, Tv)
    claimed_name = gal_names[claimed_id] if claimed_id < len(gal_names) else f"ID_{claimed_id}"
    print(f"Claimed identity: ID {claimed_id} ({claimed_name})")
    print(f"Matching score:   {score:.4f} (Tv = {Tv:.4f})")
    print(f">> {decision}")
    return score, decision


def identify_user(embedding_norm, gallery_name, database_dir, Ti):
    """
    1:N open-set identification.

    Returns:
        list of {"id", "name", "score"} above Ti, best first (empty = unknown user, rejected).
    """
    gal_embs, gal_labels, gal_names = utils.load_gallery(filename=gallery_name, gallery_path=database_dir)
    if len(gal_embs) == 0:
        raise ValueError("Gallery is empty: enroll at least one user first.")
    candidates = th_tuning.rank_candidates(embedding_norm, gal_embs, gal_labels, gal_names, Ti)
    if not candidates:
        print(f">> ACCESS DENIED: user not recognized (no score >= Ti = {Ti:.4f})")
    else:
        print(f"Candidates above Ti = {Ti:.4f}:")
        for rank, c in enumerate(candidates, 1):
            print(f"  {rank}. ID {c['id']:>3} {c['name']:<20} score={c['score']:.4f}")
        print(f">> ACCESS GRANTED: Welcome, {candidates[0]['name']}!")
    return candidates


# ---------------------------------------------------------------------------
# Session orchestration
# ---------------------------------------------------------------------------
def run_session(mode, data_dir, embedding_model, device, magface, liveness_model=None, use_liveness=True,
                user_name=None, claimed_id=None, config=None, show=False):
    """
    Runs a complete attendance session.

    Args:
        mode: "Enrollment", "Identification" or "Verification".
        data_dir: Folder with the acquired samples (at most `max_samples` are used).
        embedding_model: Name of the feature extractor.
        device: Torch device.
        magface: MagFaceEvaluator (quality gate).
        liveness_model: Trained liveness model (required if use_liveness=True).
        use_liveness: Activate/deactivate the replay-attack detector.
        user_name: Name of the new user (Enrollment).
        claimed_id: Claimed identity (Verification).
        config: Overrides of DEFAULT_CONFIG.
        show: Display the faces at each stage (notebook use).

    Returns:
        dict with the mode result and the per-stage processing times (seconds).
        Raises AcquisitionError when the sample(s) must be re-acquired.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    timings = {}
    t0 = time.perf_counter()

    # 1. Acquisition + detection (CHW crops when they go through the liveness CNN)
    faces = acquire_and_detect(data_dir, cfg["detector"], cfg["max_samples"], cnn=use_liveness)
    timings["detection"] = time.perf_counter() - t0

    # 2. Liveness (optional)
    if use_liveness:
        if liveness_model is None:
            raise ValueError("use_liveness=True requires a liveness model.")
        t = time.perf_counter()
        faces = verify_liveness(faces, liveness_model, device, cfg["batch_size"], cfg["Tf"])
        timings["liveness"] = time.perf_counter() - t
    else:
        print("\nLiveness detection is disabled.")

    # 3. Quality gate (acquisition module)
    t = time.perf_counter()
    accepted = quality_gate(faces, magface, cfg["quality_threshold"], device)
    timings["quality"] = time.perf_counter() - t
    if show:
        _show_faces(accepted, "Accepted samples (best quality first)")

    # 4. Features: all accepted samples for enrollment, only the best one for matching
    t = time.perf_counter()
    embeddings = extract_embeddings(accepted, embedding_model, device, cfg["batch_size"])
    timings["embedding"] = time.perf_counter() - t

    # 5. Operating mode
    t = time.perf_counter()
    print(f"\n{'=' * 30} {mode.upper()} {'=' * 30}")
    if mode == "Enrollment":
        if not user_name:
            raise ValueError("Enrollment requires a user name.")
        result = enroll_new_user(embeddings, user_name, cfg["gallery_name"], cfg["database_dir"])
    elif mode == "Verification":
        if claimed_id is None:
            raise ValueError("Verification requires a claimed ID.")
        result = verify_user(embeddings[0], cfg["Tv"], claimed_id, cfg["gallery_name"], cfg["database_dir"])
    elif mode == "Identification":
        result = identify_user(embeddings[0], cfg["gallery_name"], cfg["database_dir"], cfg["Ti"])
    else:
        raise ValueError(f"Invalid mode '{mode}': choose among 'Enrollment', 'Identification', 'Verification'.")
    timings["matching"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - t0

    print("\nProcessing times: " + " | ".join(f"{k}: {v:.2f}s" for k, v in timings.items()))
    return {"mode": mode, "result": result, "n_samples_used": len(accepted.images), "timings": timings}


def load_models(cfg, device, use_liveness=True):
    """Loads the MagFace quality model and (optionally) the liveness model from `cfg['model_dir']`."""
    magface = face_quality.MagFaceEvaluator(os.path.join(cfg["model_dir"], cfg["magface_model_name"]), device=device)
    liveness_model = None
    if use_liveness:
        liveness_model = liveness_detector.load_liveness_model(
            os.path.join(cfg["model_dir"], cfg["liveness_model_name"]), device)
    return magface, liveness_model


def main():
    import torch

    parser = argparse.ArgumentParser(description="Classroom attendance system - live demo")
    parser.add_argument("mode", choices=["enroll", "identify", "verify"])
    parser.add_argument("--data-dir", required=True, help="Folder with the acquired sample(s) (max 3 used)")
    parser.add_argument("--name", help="User name (enroll)")
    parser.add_argument("--claimed-id", type=int, help="Claimed identity ID (verify)")
    parser.add_argument("--no-liveness", action="store_true", help="Bypass the liveness detector")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_liveness = not args.no_liveness
    magface, liveness_model = load_models(DEFAULT_CONFIG, device, use_liveness)
    mode = {"enroll": "Enrollment", "identify": "Identification", "verify": "Verification"}[args.mode]
    try:
        run_session(mode, args.data_dir, DEFAULT_CONFIG["embedding_model"], device, magface, liveness_model,
                    use_liveness=use_liveness, user_name=args.name, claimed_id=args.claimed_id)
    except AcquisitionError as e:
        print(f"\n>> RE-ACQUISITION REQUIRED: {e}")
    except ValueError as e:
        print(f"\n>> ERROR: {e}")


if __name__ == "__main__":
    main()

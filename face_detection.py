"""
Face detection module.

Unified wrapper around three detectors (Viola-Jones, Dlib HOG+SVM, MTCNN) that
converts their outputs to a common (x, y, w, h) bounding-box format, crops the
primary face of each image and returns a normalized, fixed-size ROI.
"""
import cv2
import numpy as np
import torch
from sklearn.utils import Bunch


# ---------------------------------------------------------------------------
# Detector wrappers: each returns a list of (x, y, w, h) boxes
# ---------------------------------------------------------------------------
def detect_VJ(img, gray, model):
    """Viola & Jones (OpenCV Haar cascade). Works on grayscale images."""
    return model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))


def detect_MTCNN(img, gray, model):
    """
    MTCNN (facenet-pytorch). Works on RGB images.

    Note: `MTCNN.detect()` always returns *all* detected faces (`keep_all` only
    affects the cropping API). With the default `select_largest=True` the boxes
    are sorted by area, so the first box is the largest face in the frame.
    """
    boxes, _ = model.detect(img)
    if boxes is None:
        return []
    return [(int(x1), int(y1), int(x2 - x1), int(y2 - y1)) for x1, y1, x2, y2 in boxes]


def detect_Dlib(img, gray, model):
    """Dlib frontal face detector (HOG + linear SVM). Works on grayscale images."""
    rects = model(gray, 0)
    return [(r.left(), r.top(), r.width(), r.height()) for r in rects]


def build_detector(method):
    """Instantiates the requested detector. Heavy dependencies are imported lazily."""
    if method == 'VJ':
        model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return model, detect_VJ
    if method == 'MTCNN':
        from facenet_pytorch import MTCNN
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return MTCNN(keep_all=False, min_face_size=20, device=device), detect_MTCNN
    if method == 'Dlib':
        import dlib  # optional dependency, only needed for the detector comparison
        return dlib.get_frontal_face_detector(), detect_Dlib
    raise ValueError(f"Unknown method: {method}. Choose among 'VJ', 'MTCNN', 'Dlib'.")


def _clip_box(x, y, w, h, img_w, img_h):
    """Clips a (x, y, w, h) box to the image borders and returns (x1, y1, x2, y2)."""
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def detect_faces(unprocessed_dataset, method='MTCNN', target_names=None, verbose=False,
                 detected_face_size=(224, 224), cnn=False):
    """
    Loads every image of a dataset of FILE PATHS, detects faces and keeps the
    primary face of each image (the first box returned by the detector).

    Args:
        unprocessed_dataset: Bunch with `.images` (file paths) and `.target` (labels).
        method: 'VJ', 'MTCNN' or 'Dlib'.
        target_names: Optional array of class names (defaults to the dataset's).
        verbose: Print progress and the reason of every dropped image.
        detected_face_size: Output size of the face ROI.
        cnn: If True, faces are returned in CHW format (CNN input), otherwise HWC.

    Returns:
        Bunch with:
          .images       -> object array of float32 faces in [0, 1]
          .target       -> labels of the kept images
          .paths        -> file paths of the kept images (aligned with .images)
          .kept_indices -> indices of the kept images in the input dataset
          .target_names -> class names
        Images where no face is found are dropped, therefore `.paths` must be
        used (instead of the input paths) whenever faces and files are matched.
    """
    if target_names is None:
        target_names = getattr(unprocessed_dataset, 'target_names', None)

    if verbose:
        print(f"\nStarting face detection using {method} (output size {detected_face_size})")
    model, detect_func = build_detector(method)

    processed_images, processed_targets, kept_paths, kept_indices = [], [], [], []
    dropped = {"load_failed": 0, "no_face": 0, "empty_roi": 0}
    total_images = len(unprocessed_dataset.images)

    for i in range(total_images):
        img_path = unprocessed_dataset.images[i]
        target_id = unprocessed_dataset.target[i]

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            dropped["load_failed"] += 1
            if verbose: print(f"DROPPED [Load Failed]: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        faces = detect_func(img_rgb, img_gray, model)
        if len(faces) == 0:
            dropped["no_face"] += 1
            if verbose: print(f"DROPPED [No Face Detected]: {img_path}")
            continue

        x1, y1, x2, y2 = _clip_box(*faces[0], img_rgb.shape[1], img_rgb.shape[0])
        roi = img_rgb[y1:y2, x1:x2]
        if roi.size == 0:
            dropped["empty_roi"] += 1
            if verbose: print(f"DROPPED [Empty ROI]: {img_path}")
            continue

        face = cv2.resize(roi, detected_face_size).astype(np.float32) / 255.0
        processed_images.append(np.transpose(face, (2, 0, 1)) if cnn else face)
        processed_targets.append(target_id)
        kept_paths.append(img_path)
        kept_indices.append(i)

    images_arr = np.empty(len(processed_images), dtype=object)
    for k, face in enumerate(processed_images):
        images_arr[k] = face

    if verbose:
        print(f"Dropped images summary: {dropped}")
        print(f"Kept {len(processed_images)} out of {total_images} images.")

    return Bunch(
        images=images_arr,
        target=np.array(processed_targets),
        paths=np.array(kept_paths, dtype=object),
        kept_indices=np.array(kept_indices, dtype=int),
        target_names=target_names,
    )

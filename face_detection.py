import cv2
import numpy as np
from facenet_pytorch import MTCNN as MTCNN_PyTorch  # PyTorch version instead of TensorFlow
import dlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import Bunch
import torch
from IPython.display import display
from PIL import Image
import torch



# WRAPPERS FOR DIFFERENT FACE DETECTION LIBRARIES
# These wrappers will take the raw image and return a standardized list of bounding boxes (x, y, w, h)


def detect_VJ(img, gray, model):
    """Wrapper for Viola & Jones."""
    # VJ uses grayscale
    return model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))



def detect_MTCNN(img, gray, model):
    """Wrapper for MTCNN (facenet_pytorch version)."""
    # facenet_pytorch MTCNN expects RGB images (already in RGB from caller)
    # model.detect returns (boxes, probs) where boxes is Nx4 array or None
    # Convert numpy array to PIL Image
    img_pil = Image.fromarray(img)
    boxes, probs = model.detect(img)
    if boxes is None:
        return []
    # Convert from [x1, y1, x2, y2] to [x, y, w, h] format
    result = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        result.append((x, y, w, h))
    return result


def detect_Dlib(img, gray, model):
    """Wrapper for Dlib."""
    # Dlib uses grayscale
    rects = model(gray, 0)
    # Convert dlib objects to tuples (x, y, w, h)
    return [(r.left(), r.top(), r.width(), r.height()) for r in rects]




def detect_faces(unprocessed_dataset, method='MTCNN', target_names=None, verbose=False, detected_face_size=(224, 224), cnn=False):
    """
    Takes a dataset (Bunch object) containing file paths.
    For each image path, it loads the image, detects faces and selects the BEST face found
    (with keep_all=False, so at most one face is returned).
    Args:
        unprocessed_dataset: Bunch object with 'images' as file paths and 'target' as labels.
        method: Face detection method ('VJ', 'MTCNN', 'Dlib').
        target_names: Optional list of target names.
        verbose: If True, prints progress and dropped image reasons.
        detected_face_size: Tuple specifying the size to which detected faces are resized.
        cnn: If True, returns images in CHW format for CNN input; otherwise HWC format.
    """
    if verbose:print(f"\nStarting face detection using {method} method\nRunning with image_size={detected_face_size}")
    
    
    # Resolve target_names
    if target_names is None:
        target_names = getattr(unprocessed_dataset, 'target_names', None)
    
    # 1. Initialize Models 
    if verbose:print(f"Initializing {method} detector...")
    if method == 'VJ':
        model = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
        detect_func = detect_VJ
    elif method == 'MTCNN':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        from facenet_pytorch import MTCNN as MTCNN_PyTorch
        # keep_all=False ensures MTCNN selects the best face internally
        model = MTCNN_PyTorch(keep_all=False, min_face_size=20, device=device)
        detect_func = detect_MTCNN
    elif method == 'Dlib':
        model = dlib.get_frontal_face_detector()
        detect_func = detect_Dlib
    else:
        raise ValueError(f"Unknown method: {method}")

    # 2. Storage
    processed_images = []
    processed_targets = []
    dropped_counts = {"No Face": 0}
    total_images = len(unprocessed_dataset.images)
    if verbose: print(f"\nProcessing {total_images} images...")
    
    # 3. Iterate
    for i in range(total_images):
        # Load Image from Path 
        img_path = unprocessed_dataset.images[i]
        target_id = unprocessed_dataset.target[i]
        person_name = target_names[target_id] if target_names is not None else f"ID_{target_id}"

        # Load from disk
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                if verbose: print(f"DROPPED [Load Failed]: {img_path}")
                dropped_counts["No Face"] += 1
                continue
            img_input = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            if verbose: print(f"DROPPED [Read Error]: {img_path} ({e})")
            dropped_counts["No Face"] += 1
            continue
        
        # Convert to 0-255 uint8
        if img_input.dtype == np.float64 or img_input.dtype == np.float32 or img_input.max() <= 1.0:
            img_uint8 = (img_input * 255).astype(np.uint8)
        else:
            img_uint8 = img_input.astype(np.uint8)
            
        # Ensure RGB/Gray formats
        if img_uint8.ndim == 2:
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
            img_gray = img_uint8
        else:
            img_rgb = img_uint8
            img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            
        # Detect Faces
        faces = detect_func(img_rgb, img_gray, model)
        
        
        # With keep_all=False, 'faces' contains at most 1 element (the best one)
        if len(faces) > 0:
            # Take the first (and only) face directly
            (x, y, w, h) = faces[0]
            # Boundary checks
            x, y = max(0, x), max(0, y)
            w = min(w, img_uint8.shape[1] - x)
            h = min(h, img_uint8.shape[0] - y)
            # Crop
            best_face_roi = img_rgb[y:y+h, x:x+w]
            # Save (Resize + Transpose)
            if best_face_roi is not None and best_face_roi.size > 0:
                # Resize to detected_face_size
                best_face_roi = cv2.resize(best_face_roi, detected_face_size)
                # Normalize (0-1)
                face_roi_float = best_face_roi.astype(np.float32) / 255.0
                # Transpose HWC -> CHW
                if cnn:
                    face_roi_tensor = np.transpose(face_roi_float, (2, 0, 1)) 
                    processed_images.append(face_roi_tensor)
                else:
                    processed_images.append(face_roi_float)
                processed_targets.append(target_id)
            else:
                if verbose: print(f"DROPPED [Empty ROI]: Image {i} of {person_name} at dir {img_path}")
                dropped_counts["No Face"] += 1
        else:
            # No face found
            if verbose: print(f"DROPPED [No Face Detected]: Image {i} of {person_name} at dir {img_path}")
            dropped_counts["No Face"] += 1

    # 4. Package
    n_kept = len(processed_images)
    processed_images_arr = np.empty(n_kept, dtype=object)
    for k in range(n_kept):
        processed_images_arr[k] = processed_images[k]

    processed_dataset = Bunch(
        images=processed_images_arr, 
        target=np.array(processed_targets),
        target_names=target_names
    )
    if verbose:
        print(f"Dropped images summary: {dropped_counts}") 
        print(f"Kept {n_kept} out of {total_images} images.")
    return processed_dataset
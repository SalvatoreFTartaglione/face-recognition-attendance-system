import time
import numpy as np
import os
import cv2
import joblib
from sklearn.preprocessing import normalize
from sklearn.utils import Bunch
import face_detection as detector
import utilities as utils
import threshold_tuning as th_tuning
import liveness_detector as liveness_detector
import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


# Custom libraries imports
import face_detection as detector
import utilities as utils
import threshold_tuning as th_tuning
import liveness_detector as liveness_detector
import face_quality 


def verify_liveness(data_dir, detector_method, liveness_model, device, batch_size, Tf):
    """  
    Verify liveness of detected faces in the given data directory using the provided liveness detection model.
    Removes non-live faces from the dataset.
    Parameters:
    - data_dir (str): Directory containing the images to be processed.
    - detector_method (object): Face detection method to be used.
    - liveness_model (torch.nn.Module): Pre-trained liveness detection model.
    - device (torch.device): Device to run the model on (CPU or GPU).
    - batch_size (int): Batch size for DataLoader.
    - Tf (float): Threshold for liveness classification.
    Returns:
    - filtered_test (Bunch): Bunch object containing only live faces after filtering.
    - processd_filtered_test (Bunch): Bunch object containing processed data of live faces.
    """
    
    #########################
    # REMOVE NON LIVE FACES #
    #########################
    # 1. Download/Load the paths (evaluation needs just the paths)
    test_data_paths = utils.download_demo_data_paths(data_dir)
    # 2. Process/Detect Faces 
    processed_test = detector.detect_faces(test_data_paths, method=detector_method, verbose=False, cnn=True)
    if processed_test.images.shape[0] == 0:
            print("No faces detected. Exiting the system.")
            sys.exit()
    # 3. Wrap & Load
    test_ds = liveness_detector.LivenessDatasetWrapper(processed_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    # 4. Filter: Remove Fake Faces
    processd_filtered_test, filtered_test = liveness_detector.filter_non_live_faces(processed_test, test_data_paths, test_loader, liveness_model, device, Tf, verbose=True)
    if processd_filtered_test.images.shape[0] == 0:
        print("No live faces detected. Exiting the system.")
        sys.exit()
    # 5. Visualize only live faces
    print(f"\n{'='*50}  LIVE FACES DETECTED {'='*50}\n")
    #utils.visualize_detected_faces(processd_filtered_test, n_cols=processd_filtered_test.images.shape[0])
    
    return processd_filtered_test, filtered_test





def check_faces_and_extract_embedding(processed_filtered_test, filtered_test, model, device, batch_size, magface, img_quality_threshold):
    """
    Detects faces in the input images and extracts their embeddings using the specified model.

    Parameters:
    - filtered_test: The input images after liveness filtering.
    - detector_method: The face detection method to use.
    - model: The face embedding model.
    - device: The device to run the computations on (CPU or GPU).
    - batch_size: The batch size for processing.
    - magface: The MagFace quality assessment utility.
    - img_quality_threshold: The minimum quality score to consider a face valid using magface.
    Returns:
    - embedding_norm: The normalized embeddings of the detected faces.
    """
    utils.visualize_detected_faces(processed_filtered_test, n_cols=processed_filtered_test.images.shape[0])
    # 1. Select the single Best Quality Face
    processed_filtered_test = face_quality.quality_check_magface(magface, filtered_test, processed_filtered_test, verbose=False, device=device, img_quality_threshold=img_quality_threshold)
    
    # 2. Visualize the selected face
    face = np.array(processed_filtered_test['images'][0])
    face = face.astype(np.float32)
    plt.imshow(face)
    plt.axis('off')
    plt.show()
    # 4. Preprocess & Extract Embedding
    embedding = utils.preprocess_and_embed(processed_filtered_test, model, device, batch_size, verbose=False)
    # 5 Normalize the embedding
    embedding_norm = normalize(embedding, norm='l2')
    
    return embedding_norm




def enroll_new_user(new_embedding, new_name, gallery_path="gallery.npy", database_dir="./database/gallery"):
    """
    Loads the existing .npy gallery, appends the new user, and saves it back.
    If no gallery exists, creates a new one with the first user. 
    Parameters:
    - new_embedding (np.ndarray): The normalized embedding of the new user to enroll.
    - new_name (str): The name of the new user to enroll.
    - gallery_path (str): The filename of the gallery file.
    - database_dir (str): The directory where the gallery is stored.
    Returns:
    - None
    """
    print(f"\n{'='*50}  ENROLLMENT MODE {'='*50}\n")    
    full_path = os.path.join(database_dir, gallery_path)
    # Ensure directory exists
    os.makedirs(database_dir, exist_ok=True)
    # 1. Load Existing or Create New
    if os.path.exists(full_path):
        # Load using the new dictionary-based loader
        try:
            current_embs, current_labels, current_names = utils.load_gallery(gallery_path, database_dir)
        except Exception as e:
            print(f"Error loading gallery: {e}")
            return

        # Check if user already exists
        if new_name in current_names:
            # Find the ID of the existing user
            existing_label = np.where(current_names == new_name)[0][0]
            raise Exception(f"User '{new_name}' already exists (ID: {existing_label}).\nENROLLMENT SKIPPED (Duplicate).")
            
        else:
            # Create new ID
            new_label = len(current_names)
            # Append new name
            current_names = np.append(current_names, new_name)
            print(f"New User '{new_name}' assigned ID: {new_label}.")
            
        # 2. Append New Data
        # Ensure new embedding is (1, 512)
        new_embedding = new_embedding.reshape(1, -1)
        # Stack embeddings
        updated_embs = np.vstack((current_embs, new_embedding))
        # Append label
        updated_labels = np.append(current_labels, new_label)
        # 3. Save back to disk
        utils.create_gallery(updated_embs, updated_labels, current_names, filename=gallery_path, database_dir=database_dir, verbose=False)
        print(f"Enrollment Completed for '{new_name}'.")
        
    else:
        # No gallery exists - create a new one with the first user
        print(f"No existing gallery found at {full_path}.")
        print(f"Creating new gallery with first user '{new_name}'...")
        new_embedding = new_embedding.reshape(1, -1)  # (1, 512)
        new_labels = np.array([0])      # First user gets ID 0
        new_names = np.array([new_name])
        utils.create_gallery(new_embedding, new_labels, new_names, filename=gallery_path, database_dir=database_dir, verbose=False)    
        print(f"Enrollment Complete. New gallery created.")





def verify_user(embedding_norm, Tv, claimed_id, gallery_name, database_dir):
    """
    Verifies the user by comparing the provided embedding against the gallery.
    Parameters:
    - embedding_norm (np.ndarray): The normalized embedding of the user to verify.
    - Tv (float): The verification threshold.
    - claimed_id (int): The ID claimed by the user.
    - gallery_name (str): The name of the gallery file.
    - database_dir (str): The directory where the gallery is stored.
    Returns:
    - None
    """
    print(f"\n{'='*50}  VERIFICATION MODE {'='*50}\n")
    # 0. Load embedding from gallery
    gal_embs, gal_labels, gal_names = utils.load_gallery(filename=gallery_name, gallery_path=database_dir)
    if len(gal_embs) == 0:
        print("Error: Gallery is empty. Cannot perform identification.")
        sys.exit()
    # 1. Verify User
    _, _ = th_tuning.compute_far_frr_verification_on_probes(Tv, embedding_norm, [claimed_id], gal_embs, gal_labels, live_demo=True)
    
    
    
    
    
    
def identify_user(embedding_norm, gallery_name, database_dir, Ti, verbose=False):
    print(f"\n{'='*50}  IDENTIFICATION MODE {'='*50}\n")   
    # 0. Load embedding from gallery
    gal_embs, gal_labels, gal_names = utils.load_gallery(filename=gallery_name, gallery_path=database_dir)
    if len(gal_embs) == 0:
        print("Error: Gallery is empty. Cannot perform identification.")
        sys.exit()
    # 1. Identify User
    result, name, dist = th_tuning.identify_user(embedding_norm, gal_embs, gal_labels, gal_names, threshold=Ti)
    # 2. Display Result
    if verbose:
        print(f"Closest Match: {name}")
        print(f"Similarity:      {dist:.4f}")
        print(f"Threshold:     {Ti:.4f}")
    if result is False:
        print(">> ACCESS DENIED User not recognized")
    else:
        print(f">> ACCESS GRANTED: Welcome, {name}!")
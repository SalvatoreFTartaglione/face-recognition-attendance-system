# Model weights

Model weights are not versioned in this repository. Place the following files in this folder:

| File | Used by | Where to get it |
|------|---------|-----------------|
| `magface_iresnet50_MS1MV2_ddp_fp32.pth` | Face quality assessment (`face_quality.py`) | Official MagFace repository, *Model Zoo* → iResNet50 trained on MS1MV2: <https://github.com/IrvingMeng/MagFace> |
| `liveness_model.pth` | Replay-attack detector (`liveness_detector.py`) | **Releases** page of this repository, or retrain it with `01_face_authentication.ipynb` (*Liveness Detector Training*) |

The InceptionResnetV1-VGGFace2 recognition model and MTCNN are downloaded automatically by
[`facenet-pytorch`](https://github.com/timesler/facenet-pytorch) the first time they are used.

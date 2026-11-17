# Model weights

Model weights are not versioned in this repository. Place the following files in this folder:

| File | Used by | Where to get it |
|------|---------|-----------------|
| `magface_iresnet50_MS1MV2_ddp_fp32.pth` | Face quality assessment (`face_quality.py`) | Official MagFace repository, *Model Zoo* → iResNet50 trained on MS1MV2: <https://github.com/IrvingMeng/MagFace> |
| `liveness_model.pth` | Replay-attack detector (`liveness_detector.py`) | Train it with `notebooks/01_face_authentication.ipynb` (*Liveness Detector Training*) on a real/replayed face dataset organised as described in the main README |

The liveness weights are not distributed because they were trained on the faces of the dataset subjects (see the
*Privacy* section of the main README). Without them the demo can still run with the liveness detector bypassed
(`--no-liveness` on the command line, `USE_LIVENESS = False` in the demo notebook).

The InceptionResnetV1-VGGFace2 recognition model and MTCNN are downloaded automatically by
[`facenet-pytorch`](https://github.com/timesler/facenet-pytorch) the first time they are used.

# CCTV Intelligence — Model Zoo Directory

This directory stores pretrained neural model checkpoints and ONNX computational graphs used by the multi-modal perception stack.

## Download Models via GitHub

Model weights are not committed to Git to keep the repository lightweight. To download or verify all official model checkpoints from GitHub releases and hubs, run:

```bash
python scripts/download_models.py --all
```

To view current local model status:

```bash
python scripts/download_models.py --status
```

## Model Catalog

| Model ID | File Name | Format | Source | Description |
|---|---|---|---|---|
| `yolov8n` | `yolov8n.pt` | PyTorch / TorchScript | Ultralytics GitHub Releases | Person bounding-box pedestrian detector |
| `yolov8n_pose` | `yolov8n-pose.pt` | PyTorch | Ultralytics GitHub Releases | 17-keypoint COCO skeletal pose estimator |
| `face_yunet` | `face_detection_yunet_2023mar.onnx` | ONNX | OpenCV Zoo GitHub | SOTA 5-landmark face detector (`cv2.FaceDetectorYN`) |
| `face_sface` | `face_recognition_sface_2021dec.onnx` | ONNX | OpenCV Zoo GitHub | 128-d deep facial recognition embedder (`cv2.FaceRecognizerSF`) |
| `osnet` | `osnet_x1_0_imagenet.pth` | PyTorch StateDict | KaiyangZhou / OSNet HuggingFace | 512-d Omni-Scale Person Re-Identification |
| `rtdetr` | `rtdetr-l.pt` | PyTorch | Ultralytics / RT-DETR GitHub | High-accuracy real-time detection transformer |

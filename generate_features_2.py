import os
import cv2
import numpy as np
import torch
import glob
from datetime import datetime
import sys
import os

import yaml

from face_sdk.core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "face_sdk")))

from face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from face_sdk.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from face_sdk.core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from face_sdk.core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from face_sdk.core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from face_sdk.core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler

MODEL_ROOT = os.path.join("face_sdk", "models")
MODEL_CATEGORY = "face_recognition"
MODEL_NAME = "face_recognition_2.0"
SAMPLES_DIR = "face_samples_2"
FEATURES_DIR = "face_features"
DEBUG_DIR = "debug_dir"

recLoader = FaceRecModelLoader(MODEL_ROOT, MODEL_CATEGORY, MODEL_NAME)
recModel, rec_cfg = recLoader.load_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
recHandler = FaceRecModelHandler(recModel, device, rec_cfg)

with open('face_sdk/config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)

scene = 'non-mask'
model_path = os.path.join("face_sdk", "models")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

faceDetLoader = FaceDetModelLoader(model_path, 'face_detection', model_conf[scene]['face_detection'])
faceDetModel, det_cfg = faceDetLoader.load_model()
faceDetHandler = FaceDetModelHandler(faceDetModel, device, det_cfg)

alignLoader = FaceAlignModelLoader(model_path, 'face_alignment', model_conf[scene]['face_alignment'])
alignModel, align_cfg = alignLoader.load_model()
alignHandler = FaceAlignModelHandler(alignModel, device, align_cfg)

recLoader = FaceRecModelLoader(model_path, 'face_recognition', model_conf[scene]['face_recognition'])
recModel, rec_cfg = recLoader.load_model()
recHandler = FaceRecModelHandler(recModel, device, rec_cfg)

cropper = FaceRecImageCropper()

for person_dir in os.listdir(SAMPLES_DIR):
    full_dir = os.path.join(SAMPLES_DIR, person_dir)
    if not os.path.isdir(full_dir):
        continue

    print(f"[INFO] Przetwarzanie: {person_dir}")
    person_features = []

    frame = cv2.imread(glob.glob(os.path.join(full_dir, "*.jpg"))[0], cv2.IMREAD_COLOR)
    dets = faceDetHandler.inference_on_image(frame)

    for i in range(dets.shape[0]):
        box = dets[i]

        # Get facial landmarks for alignment
        landmarks = alignHandler.inference_on_image(frame, box)
        landmark_list = []
        for (x, y) in landmarks.astype(np.int32):
            landmark_list.extend((x, y))

        cropped = cropper.crop_image_by_mat(frame, landmark_list)
        resized = cv2.resize(cropped, (rec_cfg['input_width'], rec_cfg['input_height']))

        # Generate feature vector for the face
        feature = recHandler.inference_on_image(resized)

        output_dir = os.path.join(FEATURES_DIR, person_dir)
        os.makedirs(output_dir, exist_ok=True)
        feature_path = os.path.join(output_dir, "feature.npy")
        np.save(feature_path, feature)
        print(f"[OK] Zapisano cechy do {feature_path}")


print("[DONE] Wygenerowano wszystkie cechy.")

import os
import cv2
import numpy as np
import torch
import glob
from datetime import datetime
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "face_sdk")))
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler


MODEL_ROOT = os.path.join("face_sdk", "models")
MODEL_CATEGORY = "face_recognition"
MODEL_NAME = "face_recognition_2.0"
SAMPLES_DIR = "face_samples"
FEATURES_DIR = "face_features"

recLoader = FaceRecModelLoader(MODEL_ROOT, MODEL_CATEGORY, MODEL_NAME)
recModel, rec_cfg = recLoader.load_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
recHandler = FaceRecModelHandler(recModel, device, rec_cfg)

for person_dir in os.listdir(SAMPLES_DIR):
    full_dir = os.path.join(SAMPLES_DIR, person_dir)
    if not os.path.isdir(full_dir):
        continue

    print(f"[INFO] Przetwarzanie: {person_dir}")
    person_features = []

    for img_path in glob.glob(os.path.join(full_dir, "*.jpg")):
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_resized = cv2.resize(img, (rec_cfg['input_width'], rec_cfg['input_height']))
        try:
            # Perform inference to extract face features
            feature = recHandler.inference_on_image(img_resized)
            person_features.append(feature)
        except Exception as e:
            print(f"[WARN] Nieudane przetwarzanie {img_path}: {e}")
            continue

    if not person_features:
        print(f"[WARN] Brak cech dla {person_dir}")
        continue
    # Average the features to get a representative embedding for the person
    avg_feature = np.mean(person_features, axis=0)


    output_dir = os.path.join(FEATURES_DIR, person_dir)
    os.makedirs(output_dir, exist_ok=True)
    feature_path = os.path.join(output_dir, "feature.npy")
    np.save(feature_path, avg_feature)
    print(f"[OK] Zapisano cechy do {feature_path}")

print("[DONE] Wygenerowano wszystkie cechy.")

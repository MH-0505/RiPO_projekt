import os
import cv2
import numpy as np
import torch
import glob
import sys

# Ścieżka do biblioteki SDK
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "face_sdk")))
from face_sdk.core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from face_sdk.core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler

# Konfiguracja
MODEL_ROOT = os.path.join("face_sdk", "models")
MODEL_CATEGORY = "face_recognition"
MODEL_NAME = "face_recognition_2.0"
SAMPLES_DIR = "face_samples"
FEATURES_DIR = "face_features"

# Wczytanie modelu
recLoader = FaceRecModelLoader(MODEL_ROOT, MODEL_CATEGORY, MODEL_NAME)
recModel, rec_cfg = recLoader.load_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
recHandler = FaceRecModelHandler(recModel, device, rec_cfg)

# Przetwarzanie zdjęć osób
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

        try:
            # Skalowanie obrazu do wymiarów wejściowych modelu
            img_resized = cv2.resize(img, (rec_cfg['input_width'], rec_cfg['input_height']))

            # Ekstrakcja cech
            feature = recHandler.inference_on_image(img_resized)
            feature = feature / np.linalg.norm(feature)
            person_features.append(feature)
        except Exception as e:
            print(f"[WARN] Nieudane przetwarzanie {img_path}: {e}")
            continue

    if not person_features:
        print(f"[WARN] Brak cech dla {person_dir}")
        continue

    # Średnia cech danej osoby
    avg_feature = np.mean(person_features, axis=0)
    avg_feature = avg_feature / np.linalg.norm(avg_feature)

    output_dir = os.path.join(FEATURES_DIR, person_dir)
    os.makedirs(output_dir, exist_ok=True)
    feature_path = os.path.join(output_dir, "feature.npy")
    np.save(feature_path, avg_feature)
    print(f"[OK] Zapisano cechy do {feature_path}")

print("[DONE] Wygenerowano wszystkie cechy.")

import cv2
import numpy as np
import os
import glob
import yaml
import torch
import logging
import logging.config



from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'face_sdk'))
import os
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'face_sdk/config/logging.conf'))

logging.config.fileConfig(config_path)



from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper


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

# Load known face features from previously saved .npy files
known_features = {}
for person_dir in glob.glob("face_features/*"):
    person_name = os.path.basename(person_dir)
    features = []
    for feature_file in glob.glob(os.path.join(person_dir, "*.npy")):
        features.append(np.load(feature_file))
    known_features[person_name] = features

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='camera')  # camera, video, ip
parser.add_argument('--path', type=str, default='')  # path to video file or ip stream
args = parser.parse_args()

if args.source == "camera":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
elif args.source in ["video", "ip"]:
    cap = cv2.VideoCapture(args.path)
else:
    print("[BŁĄD] Nieznane źródło:", args.source)
    exit()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
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

        # Compare with known features
        best_match = None
        best_score = -1
        for name, features in known_features.items():
            for known in features:
                score = np.dot(feature, known)
                if score > best_score:
                    best_score = score
                    best_match = name

        label = "Unknown"

        x1, y1, x2, y2 = box[:4].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{best_match} ({best_score * 100:.1f}%)" if best_match else "Unknown"
        cv2.putText(frame, label, (x1, x2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Rozpoznawanie twarzy", frame)
        if cv2.getWindowProperty("Rozpoznawanie twarzy", cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

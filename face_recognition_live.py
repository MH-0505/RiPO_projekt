import cv2
import numpy as np
import os
import glob
import yaml
import torch
import logging
import logging.config

from face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from face_sdk.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from face_sdk.core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from face_sdk.core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from face_sdk.core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from face_sdk.core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler
from face_sdk.core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

from datetime import datetime
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'face_sdk'))
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'face_sdk/config/logging.conf'))

logging.config.fileConfig(config_path)


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

interval = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    dets = faceDetHandler.inference_on_image(frame)

    if interval == 0 and False: # roboczo tego używałem do zbierania próbek ( raczej do wywalenia )
        for i in range(dets.shape[0]):
            box = dets[i]
            x1, y1, x2, y2 = box[:4].astype(int)

            face_img = frame[y1:y2, x1:x2]
            output_dir = os.path.join("face_samples", '2_dnn')
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = os.path.join(output_dir, f"face_{timestamp}.jpg")
            cv2.imwrite(filename, face_img)
    interval = interval + 1
    interval = interval % 20


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
        best_score = 0

        for name, features in known_features.items():
            for known in features:
                score = np.dot(feature, known)
                if score > best_score:
                    best_score = score
                    best_match = name

        x1, y1, x2, y2 = box[:4].astype(int)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{best_match} ({best_score * 100:.1f}%)" if best_match else "Unknown"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Rozpoznawanie twarzy", frame)

        key = cv2.waitKey(25) & 0xFF
        if (key == 27 and key == ord('q')) or cv2.getWindowProperty("Rozpoznawanie twarzy", cv2.WND_PROP_VISIBLE) < 1:
            cap.release()
            cv2.destroyAllWindows()
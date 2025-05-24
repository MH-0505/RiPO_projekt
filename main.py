import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'face_sdk'))

from tkinter import messagebox
import cv2 as cv
import numpy as np
import threading
import tkinter as tk
from tkinter import *
import glob
import HaarCascadesFD
from datetime import datetime

import yaml
import torch
from face_sdk.core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from face_sdk.core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler
from face_sdk.core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from face_sdk.core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from face_sdk.core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

window = tk.Tk()
selected_video = StringVar()
detect_faces = StringVar()
detection_method = StringVar()
selected_source = StringVar()
camera_ip_url = StringVar()
subject_name = StringVar()
operation_mode = StringVar()

haar_casc_interval = IntVar()

DEBUG_DIR = "debug_dir"

def main():
    init_window()


def init_window():
    global selected_video, window, detect_faces, selected_source, camera_ip_url

    window.title('RiPO - konfigurator')

    top_bar = tk.Frame(window)
    top_bar.pack(side=tk.TOP, fill=tk.X, pady=10, padx=10)

    videos = glob.glob('test_videos/*')

    if not videos:
        messagebox.showerror("Błąd", "Brak nagrań w folderze test_videos")
        window.quit()
        return
    mode_frame = tk.LabelFrame(window, bd=2, relief="groove", text="Tryb działania")
    mode_frame.pack(side=tk.TOP, fill=tk.X, pady=10, padx=10)

    operation_mode.set("Zbieranie zdjęć wzorcowych")

    tk.Label(mode_frame, text="Wybierz tryb:").pack(side=tk.LEFT, padx=5)
    OptionMenu(mode_frame, operation_mode,
               "Zbieranie zdjęć wzorcowych",
               "Wykrywanie twarzy",
               "Rozpoznawanie osoby").pack(side=tk.LEFT, padx=5)
    selected_source.set("Plik wideo")
    selected_video.set(videos[0])
    camera_ip_url.set("rtsp://")

    tk.Label(top_bar, text="Nazwa podmiotu:").pack(padx=5, side=tk.LEFT)
    Entry(top_bar, textvariable=subject_name, width=15).pack(side=tk.LEFT)

    tk.Label(top_bar, text="Źródło obrazu:").pack(padx=5, side=tk.LEFT)
    OptionMenu(top_bar, selected_source, "Plik wideo", "Kamera", "Kamera IP").pack(padx=5, side=tk.LEFT)

    ip_frame = tk.Frame(top_bar)
    tk.Label(ip_frame, text="Adres IP kamery:").pack(side=tk.LEFT)
    Entry(ip_frame, textvariable=camera_ip_url, width=25).pack(side=tk.LEFT)

    video_frame = tk.Frame(top_bar)
    tk.Label(video_frame, text="Wybierz wideo:").pack(side=tk.LEFT)
    dropdown_list = OptionMenu(video_frame, selected_video, *videos)
    dropdown_list.pack(side=tk.LEFT)

    def update_source_fields(*args):
        source = selected_source.get()
        if source == "Kamera IP":
            ip_frame.pack(side=tk.LEFT, padx=5)
            video_frame.pack_forget()
        elif source == "Plik wideo":
            video_frame.pack(side=tk.LEFT, padx=5)
            ip_frame.pack_forget()
        else:
            ip_frame.pack_forget()
            video_frame.pack_forget()

    selected_source.trace_add("write", update_source_fields)
    update_source_fields()

    button = Button(top_bar, text='Start', command=button_pressed)
    button.pack(padx=20, side=tk.LEFT)

    gen_features_btn = Button(top_bar, text='Wygeneruj cechy', command=generate_features)
    gen_features_btn.pack(padx=5, side=tk.LEFT)

    checkbox = Checkbutton(top_bar, text='Wykrywaj twarz', variable=detect_faces)
    checkbox.select()
    checkbox.pack(padx=5, side=tk.LEFT)

    haar_casc_bar = tk.LabelFrame(window, bd=2, relief="groove", text="Haar Cascades")
    haar_casc_bar.pack(side=tk.TOP, fill=tk.X, pady=10, padx=10)
    radio1 = Radiobutton(haar_casc_bar, text="W użyciu", variable=detection_method, value='0')
    radio1.select()
    radio1.pack(padx=5, side=tk.LEFT)
    tk.Label(haar_casc_bar, text="Interwał:").pack(padx=5, side=tk.LEFT)
    scale1 = Scale(haar_casc_bar, variable=haar_casc_interval, from_=0, to=30, orient=HORIZONTAL, showvalue=True)
    scale1.pack(padx=5, side=tk.LEFT)
    scale1.set(10)

    hog_bar = tk.LabelFrame(window, bd=2, relief="groove", text="HOG + SVM")
    hog_bar.pack(side=tk.TOP, fill=tk.X, pady=10, padx=10)
    radio2 = Radiobutton(hog_bar, text="W użyciu", variable=detection_method, value='1')
    radio2.pack(padx=5, side=tk.LEFT)

    dnn_bar = tk.LabelFrame(window, bd=2, relief="groove", text="DNN")
    dnn_bar.pack(side=tk.TOP, fill=tk.X, pady=10, padx=10)
    radio3 = Radiobutton(dnn_bar, text="W użyciu", variable=detection_method, value='2')
    radio3.pack(padx=5, side=tk.LEFT)

    window.mainloop()

def generate_features():
    import subprocess
    try:
        subprocess.run([sys.executable, "generate_features.py"], check=True)
        messagebox.showinfo("Sukces", "Wygenerowano cechy pomyślnie.")
    except subprocess.CalledProcessError:
        messagebox.showerror("Błąd", "Nie udało się wygenerować cech.")

def button_pressed():
    if detection_method.get() == '0':
        process_video(HaarCascadesFD.HaarCascadesFD(interval=haar_casc_interval.get()+1))
    elif detection_method.get() == '1':
        messagebox.showerror("Błąd", "Metoda HOG + SVM jeszcze nie jest zaimplementowana :(")
    elif detection_method.get() == '2':
        import subprocess
        source = selected_source.get()
        try:
            if source == "Kamera":
                subprocess.Popen([sys.executable, "face_recognition_live.py", "--source", "camera"])
            elif source == "Plik wideo":
                subprocess.Popen(
                    [sys.executable, "face_recognition_live.py", "--source", "video", "--path", selected_video.get()])
            elif source == "Kamera IP":
                subprocess.Popen(
                    [sys.executable, "face_recognition_live.py", "--source", "ip", "--path", camera_ip_url.get()])
            else:
                messagebox.showerror("Błąd", "Nieznane źródło obrazu")
                return
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się uruchomić DNN:\n{e}")


def process_video(face_detector):
    source = selected_source.get()
    if source == "Plik wideo":
        cap = cv.VideoCapture(selected_video.get())
    elif source == "Kamera":
        cap = cv.VideoCapture(0)
    elif source == "Kamera IP":
        cap = cv.VideoCapture(camera_ip_url.get())
    else:
        messagebox.showerror("Błąd", "Nieznane źródło obrazu")
        return

    counter = 0
    faces = []

    if not cap.isOpened():
        messagebox.showerror("Błąd", "Nie można otworzyć pliku wideo")
        return

    if operation_mode.get() == "Rozpoznawanie osoby":
        with open('face_sdk/config/model_conf.yaml') as f:
            model_conf = yaml.load(f, Loader=yaml.FullLoader)

        scene = 'non-mask'
        model_path = 'face_sdk/models'
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        alignLoader = FaceAlignModelLoader(model_path, 'face_alignment', model_conf[scene]['face_alignment'])
        alignModel, align_cfg = alignLoader.load_model()
        alignHandler = FaceAlignModelHandler(alignModel, device, align_cfg)

        recLoader = FaceRecModelLoader(model_path, 'face_recognition', model_conf[scene]['face_recognition'])
        recModel, rec_cfg = recLoader.load_model()
        recHandler = FaceRecModelHandler(recModel, device, rec_cfg)

        cropper = FaceRecImageCropper()

        known_features = {}
        for person_dir in glob.glob("face_features/*"):
            person_name = os.path.basename(person_dir)
            features = []
            for feature_file in glob.glob(os.path.join(person_dir, "*.npy")):
                features.append(np.load(feature_file))
            known_features[person_name] = features

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if counter % face_detector.interval == 0 and detect_faces.get() == '1':
            faces = face_detector.detect_face(frame)

        counter += 1

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]

            if operation_mode.get() == "Zbieranie zdjęć wzorcowych":
                name = subject_name.get().strip()
                if not name:
                    messagebox.showerror("Błąd", "Wprowadź nazwę podmiotu przed rozpoczęciem")
                    return
                output_dir = os.path.join("face_samples", name)
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = os.path.join(output_dir, f"face_{timestamp}.jpg")
                cv.imwrite(filename, face_img)

            elif operation_mode.get() == "Rozpoznawanie osoby":
                try:
                    face_img_resized = cv.resize(face_img, (rec_cfg['input_width'], rec_cfg['input_height']))
                    feature = recHandler.inference_on_image(face_img_resized)
                    feature = feature / np.linalg.norm(feature)

                    best_match = None
                    best_score = 0
                    for name, feature_list in known_features.items():
                        for known in feature_list:
                            score = np.dot(feature, known)
                            if score > best_score:
                                best_score = score
                                best_match = name

                    label = f"{best_match} ({best_score * 100:.1f}%)" if best_match else "Unknown"
                    print(label)
                    cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except Exception as e:
                    print("Błąd rozpoznawania:", e)

            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 4)

        cv.imshow('Odtwarzanie', frame)

        if cv.waitKey(25) & (0xFF == ord('q') or cv.getWindowProperty('Odtwarzanie', cv.WND_PROP_VISIBLE) < 1):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
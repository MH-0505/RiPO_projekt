import sys
import os
from random import randint

from PIL import Image as pilImage
from PIL import ImageTk

from face_sdk.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader

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
selected_source = StringVar()
camera_ip_url = StringVar()
subject_name = StringVar()
operation_mode = StringVar()

detection_interval = IntVar()
threshold = IntVar()

DEBUG_DIR = "debug_dir"

def main():
    init_window()


def init_window():
    global selected_video, window, selected_source, camera_ip_url

    window.title('RiPO - konfigurator')
    window.geometry("460x500")
    window.resizable(False, False)

    videos = glob.glob('test_videos/*')
    if not videos:
        messagebox.showerror("Błąd", "Brak nagrań w folderze test_videos")
        window.quit()
        return

    operation_mode.set("Zbieranie zdjęć wzorcowych")
    selected_source.set("Plik wideo")
    selected_video.set(videos[0])
    camera_ip_url.set("rtsp://")

    main_frame = tk.Frame(window, padx=10, pady=10)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # tryb działania
    mode_frame = tk.LabelFrame(main_frame, text="Tryb działania", padx=10, pady=5)
    mode_frame.pack(fill=tk.X, pady=5)
    mode_frame.grid_columnconfigure(1, weight=1)
    mode_frame.grid_columnconfigure(0, minsize=160)

    tk.Label(mode_frame, text="Wybierz tryb:").grid(row=0, column=0, sticky="w")
    tk.OptionMenu(mode_frame, operation_mode,
                  "Zbieranie zdjęć wzorcowych",
                  "Identyfikacja osoby").grid(row=0, column=1, sticky="ew", padx=10)

    # Dane i źródło
    source_frame = tk.LabelFrame(main_frame, text="Dane i źródło obrazu", padx=10, pady=5)
    source_frame.pack(fill=tk.X, pady=5)
    source_frame.grid_columnconfigure(1, weight=1)
    source_frame.grid_columnconfigure(0, minsize=160)

    tk.Label(source_frame, text="Nazwa podmiotu:").grid(row=0, column=0, sticky="w", pady=5)
    tk.Entry(source_frame, textvariable=subject_name).grid(row=0, column=1, sticky="ew")

    tk.Label(source_frame, text="Źródło obrazu:").grid(row=1, column=0, sticky="w")
    tk.OptionMenu(source_frame, selected_source, "Plik wideo", "Kamera", "Kamera IP").grid(row=1, column=1, sticky="ew")

    video_frame = tk.Frame(source_frame)

    video_frame.grid_columnconfigure(1, weight=1)
    video_frame.grid_columnconfigure(0, minsize=160)
    ip_frame = tk.Frame(source_frame)
    ip_frame.grid_columnconfigure(1, weight=1)
    ip_frame.grid_columnconfigure(0, minsize=160)

    tk.Label(video_frame, text="Wybierz wideo:").grid(row=0, column=0, sticky="w")
    tk.OptionMenu(video_frame, selected_video, *videos).grid(row=0, column=1, sticky="ew")

    tk.Label(ip_frame, text="Adres IP kamery:").grid(row=0, column=0, sticky="w")
    tk.Entry(ip_frame, textvariable=camera_ip_url, width=30).grid(row=0, column=1, sticky="ew")

    def update_source_fields(*args):
        source = selected_source.get()
        if source == "Kamera IP":
            ip_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
            video_frame.grid_forget()
        elif source == "Plik wideo":
            video_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
            ip_frame.grid_forget()
        else:
            ip_frame.grid_forget()
            video_frame.grid_forget()

    selected_source.trace_add("write", update_source_fields)
    update_source_fields()

    # Parametry
    param_frame = tk.LabelFrame(main_frame, text="Parametry detekcji", padx=10, pady=5)
    param_frame.pack(fill=tk.X, pady=5)
    param_frame.grid_columnconfigure(1, weight=1)
    param_frame.grid_columnconfigure(0, minsize=160)

    tk.Label(param_frame, text="Interwał detekcji:").grid(row=0, column=0, sticky="w")
    tk.Scale(param_frame, variable=detection_interval, from_=1, to=30, orient=tk.HORIZONTAL).grid(row=0, column=1, sticky="ew")
    detection_interval.set(1)

    tk.Label(param_frame, text="Próg zgodności [%]:").grid(row=1, column=0, sticky="w")
    tk.Scale(param_frame, variable=threshold, from_=1, to=100, orient=tk.HORIZONTAL).grid(row=1, column=1, sticky="ew")
    threshold.set(50)

    # przyciski
    button_frame = tk.Frame(main_frame, pady=10)
    button_frame.pack(fill=tk.X)

    tk.Button(button_frame, text='Start', command=button_pressed, width=20).pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame, text='Wygeneruj cechy', command=generate_features, width=20).pack(side=tk.LEFT, padx=10)

    window.mainloop()


def generate_features():
    import subprocess
    try:
        subprocess.run([sys.executable, "generate_features_2.py"], check=True)
        #subprocess.run([sys.executable, "generate_features.py"], check=True)

        messagebox.showinfo("Sukces", "Wygenerowano cechy pomyślnie.")
    except subprocess.CalledProcessError:
        messagebox.showerror("Błąd", "Nie udało się wygenerować cech.")


def button_pressed():
    if operation_mode.get() == "Zbieranie zdjęć wzorcowych":
        get_face_sample()
    elif operation_mode.get() == "Identyfikacja osoby":
        import subprocess
        source = selected_source.get()
        try:
            if source == "Kamera":
                subprocess.Popen(
                    [sys.executable, "face_recognition_live.py",
                     "--source", "camera",
                     "--det-interval", str(detection_interval.get()),
                     "--det-threshold", str(threshold.get())])
            elif source == "Plik wideo":
                subprocess.Popen(
                    [sys.executable, "face_recognition_live.py",
                     "--source", "video",
                     "--path", selected_video.get(),
                     "--det-interval", str(detection_interval.get()),
                     "--det-threshold", str(threshold.get())])
            elif source == "Kamera IP":
                subprocess.Popen(
                    [sys.executable, "face_recognition_live.py",
                     "--source", "ip",
                     "--path", camera_ip_url.get(),
                     "--det-interval", str(detection_interval.get()),
                     "--det-threshold", str(threshold.get())])
            else:
                messagebox.showerror("Błąd", "Nieznane źródło obrazu")
                return
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się uruchomić DNN:\n{e}")


def compute_sharpness(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return cv.Laplacian(gray, cv.CV_64F).var()


def ask_user_about_face(face_image_np):
    result = {"choice": 'none'}

    # Konwertuj obraz (OpenCV -> PIL -> PhotoImage)
    face_image_rgb = cv.cvtColor(face_image_np, cv.COLOR_BGR2RGB)
    pil_image = pilImage.fromarray(face_image_rgb).resize((250, 250))
    tk_image = ImageTk.PhotoImage(pil_image)

    win = Toplevel()
    win.title("Znaleziono twarz")

    label = Label(win, text="Czy chcesz zapisać tę próbkę?")
    label.pack(pady=5)

    image_label = Label(win, image=tk_image)
    image_label.image = tk_image
    image_label.pack()

    def save():
        result["choice"] = "save"
        win.destroy()

    def skip():
        result["choice"] = "skip"
        win.destroy()

    def cancel():
        result["choice"] = "cancel"
        win.destroy()

    Button(win, text="Zapisz", command=save, width=20).pack(pady=5)
    Button(win, text="Wygeneruj nową", command=skip, width=20).pack(pady=5)
    Button(win, text="Anuluj", command=cancel, width=20).pack(pady=5)

    win.grab_set()
    win.wait_window()

    return result["choice"]

def get_face_sample():
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

    if not cap.isOpened():
        messagebox.showerror("Błąd", "Nie można otworzyć pliku wideo")
        return

    if operation_mode.get() == "Zbieranie zdjęć wzorcowych":
        name = subject_name.get().strip()
        if not name:
            messagebox.showerror("Błąd", "Wprowadź nazwę podmiotu przed rozpoczęciem")
            return

        # Wczytaj modele do wyrównywania
        try:
            with open('face_sdk/config/model_conf.yaml') as f:
                model_conf = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            messagebox.showerror("Błąd", "Brak pliku konfiguracyjnego modelu")
            return

        scene = 'non-mask'
        model_path = 'face_sdk/models'
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        faceDetLoader = FaceDetModelLoader(model_path, 'face_detection', model_conf[scene]['face_detection'])
        faceDetModel, det_cfg = faceDetLoader.load_model()
        faceDetHandler = FaceDetModelHandler(faceDetModel, device, det_cfg)

        alignLoader = FaceAlignModelLoader(model_path, 'face_alignment', model_conf[scene]['face_alignment'])
        alignModel, align_cfg = alignLoader.load_model()
        alignHandler = FaceAlignModelHandler(alignModel, device, align_cfg)
        cropper = FaceRecImageCropper()

        sample_index = 1

        while True:
            best_face = None
            best_sharpness = 0.0
            count = 0

            for _ in range(50):
                count += 1
                ret, frame = cap.read()
                if not ret:
                    break

                cv.imshow('Odtwarzanie', frame)

                if count % randint(1, 10) != 0:
                    continue

                dets = faceDetHandler.inference_on_image(frame)

                for i in range(dets.shape[0]):
                    box = dets[i]
                    try:
                        landmarks = alignHandler.inference_on_image(frame, box)
                        landmark_list = landmarks.astype(np.int32).flatten().tolist()
                        aligned = cropper.crop_image_by_mat(frame, landmark_list)
                        sharpness = compute_sharpness(aligned)

                        if sharpness > best_sharpness:
                            best_sharpness = sharpness
                            best_face = aligned
                    except Exception as e:
                        print(f"[Błąd] Wyrównanie nie powiodło się: {e}")

                if cv.waitKey(25) & 0xFF == ord('q'):
                    cap.release()
                    cv.destroyAllWindows()
                    return

            if best_face is not None:
                user_choice = ask_user_about_face(best_face)

                if user_choice == "save":
                    output_dir = os.path.join("face_samples_2", name)
                    os.makedirs(output_dir, exist_ok=True)
                    filename = os.path.join(output_dir, f"face_{sample_index}.jpg")
                    cv.imwrite(filename, best_face)
                    sample_index += 1
                    messagebox.showinfo("Zapisano", f"Zapisano {filename}")
                elif user_choice == "skip":
                    continue
                elif user_choice == "cancel":
                    break

                cv.destroyWindow("Najlepsza twarz")
            else:
                messagebox.showwarning("Uwaga", "Nie znaleziono dobrej twarzy — spróbuj ponownie.")
                break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
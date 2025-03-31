from tkinter import messagebox

import cv2 as cv
import numpy as np
import threading
import tkinter as tk
from tkinter import *
import glob
import HaarCascadesFD
import os
from datetime import datetime


window = tk.Tk()
selected_video = StringVar()
detect_faces = StringVar()
detection_method = StringVar()
selected_source = StringVar()
camera_ip_url = StringVar()
subject_name = StringVar()
operation_mode = StringVar()

haar_casc_interval = IntVar()


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

    operation_mode.set("Zbieranie zdjęć wzorcowych")  # domyślny tryb

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

    button = Button(top_bar, text='Odtwórz nagranie', command=button_pressed)
    button.pack(padx=20, side=tk.LEFT)


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


def button_pressed():
    if detection_method.get() == '0':
        process_video(HaarCascadesFD.HaarCascadesFD(interval=haar_casc_interval.get()+1))
    elif detection_method.get() == '1':
        messagebox.showerror("Błąd", "Metoda HOG + SVM jeszcze nie jest zaimplementowana :(")
    elif detection_method.get() == '2':
        messagebox.showerror("Błąd", "Metoda DNN jeszcze nie jest zaimplementowana :(")


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


    if operation_mode.get() == "Zbieranie zdjęć wzorcowych":
        name = subject_name.get().strip()
        if not name:
            messagebox.showerror("Błąd", "Wprowadź nazwę podmiotu przed rozpoczęciem")
            return
        output_dir = os.path.join("face_samples", name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if counter % face_detector.interval == 0 and detect_faces.get() == '1':
            faces = face_detector.detect_face(frame)

        counter += 1

        for (x, y, w, h) in faces:
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 4)
            if operation_mode.get() == "Zbieranie zdjęć wzorcowych":
                face_img = frame[y:y + h, x:x + w]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = os.path.join(output_dir, f"face_{timestamp}.jpg")
                cv.imwrite(filename, face_img)
        cv.imshow('Odtwarzanie', frame)

        if cv.waitKey(25) & (0xFF == ord('q') or cv.getWindowProperty('Odtwarzanie', cv.WND_PROP_VISIBLE) < 1):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

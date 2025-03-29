from tkinter import messagebox

import cv2 as cv
import numpy as np
import threading
import tkinter as tk
from tkinter import *
import glob
import HaarCascadesFD


window = tk.Tk()
selected_video = StringVar()
detect_faces = StringVar()
detection_method = StringVar()

haar_casc_interval = IntVar()


def main():
    init_window()


def init_window():
    global selected_video, window, detect_faces

    window.title('RiPO - konfigurator')

    top_bar = tk.Frame(window)
    top_bar.pack(side=tk.TOP, fill=tk.X, pady=10, padx=10)

    videos = glob.glob('test_videos/*')

    if not videos:
        messagebox.showerror("Błąd", "Brak nagrań w folderze test_videos")
        window.quit()
        return

    selected_video.set(videos[0])
    tk.Label(top_bar, text="Wybierz wideo:").pack(padx=5, side=tk.LEFT)
    dropdown_list = OptionMenu(top_bar, selected_video, *videos)
    dropdown_list.pack(padx=5, side=tk.LEFT)

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
    cap = cv.VideoCapture(selected_video.get())
    counter = 0
    faces = []

    if not cap.isOpened():
        messagebox.showerror("Błąd", "Nie można otworzyć pliku wideo")
        return

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if counter % face_detector.interval == 0 and detect_faces.get() == '1':
            faces = face_detector.detect_face(frame)

        counter += 1

        for (x, y, w, h) in faces:
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 4)

        cv.imshow('Odtwarzanie', frame)

        if cv.waitKey(25) & (0xFF == ord('q') or cv.getWindowProperty('Odtwarzanie', cv.WND_PROP_VISIBLE) < 1):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

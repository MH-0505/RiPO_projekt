import cv2
import numpy as np
import tkinter as tk
from tkinter import StringVar, OptionMenu, Button, messagebox
import glob

window = tk.Tk()
selected_video = StringVar()


def main():
    init_window()


def init_window():
    global selected_video, window

    window.title('RiPO projekt')
    window.geometry('500x300')

    videos = glob.glob('test_videos/*')

    if not videos:
        messagebox.showerror("Błąd", "Brak nagrań w folderze test_videos")
        window.quit()
        return

    selected_video.set(videos[0])

    tk.Label(window, text="Wybierz wideo:").pack(pady=10)
    dropdown_list = OptionMenu(window, selected_video, *videos)
    dropdown_list.pack(pady=5)

    button = Button(window, text='Uruchom nagranie', command=button_pressed)
    button.pack(pady=20)

    window.mainloop()


def button_pressed():
    play_video()


def play_video():
    cap = cv2.VideoCapture(selected_video.get())

    if not cap.isOpened():
        messagebox.showerror("Błąd", "Nie można otworzyć pliku wideo")
        return

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow('Odtwarzanie', frame)

        if cv2.waitKey(25) & (0xFF == ord('q') or cv2.getWindowProperty('Odtwarzanie', cv2.WND_PROP_VISIBLE) < 1):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

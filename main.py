import cv2
import numpy as np


def main():
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture('test_videos/test2.mp4')

    if not cap.isOpened():
        print('camera is not opened')
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print('video end')
            break

        cv2.imshow('frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):   # Q - kończy wyświetlanie nagrania
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# Imports
import datetime
import glob
import os

from tqdm import tqdm
import numpy as np
import cv2

# Constants
PATH_DATA = "./data"
PATH_CONFIG = "./config"
PATH_DATASET = os.path.join(PATH_DATA, "compilation.h5")
PATH_HAAR_EYES = os.path.join(PATH_CONFIG, "haarcascade_eye.xml")

SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 900
TARGET_SIZE = 5

DATA_EYES_RESOLUTION = (100, 100)
DATA_NUMBER_COLLECT = 60


def collect_data():
    """Routine to collect new data."""
    # Set the series name
    series_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Checking for conflict with series name
    list_conflict = glob.glob(os.path.join(PATH_DATA, series_name + "*.png"))
    if len(list_conflict) > 0:
        answer = input("Conflict detected with this series name.\nDo you want to delete "
                       + "{} items? [y/N]".format(len(list_conflict)))

        # Delete the conflicting files if 'y' or 'Y' is inputed
        if answer == 'y' or answer == 'Y':
            # if True:
            for f in list_conflict:
                os.remove(f)
            os.remove(os.path.join(PATH_DATA, series_name + "_labels.csv"))

        # Abort the program otherwise.
        else:
            print("Acquisition aborted")
            return -1

    # Opening video capture
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise Exception("Could not open webcam")

    # Opening display window
    cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Display the gaze instructions and wait 1 second.
    img = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, 'TRACK THE TARGET', (50, 300), font, 4, (255, 255, 255), 2)
    cv2.imshow('Frame', img)
    cv2.waitKey(1000)

    # Loop to capture frames
    labels = []
    for ind in range(DATA_NUMBER_COLLECT):
        # Set the x and y coordinates
        x_nor, y_nor = np.random.rand(), np.random.rand()
        x_pix, y_pix = int(x_nor * SCREEN_WIDTH), int(y_nor * SCREEN_HEIGHT)

        # Draw the target
        img = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), np.uint8)
        cv2.rectangle(img,
                      (x_pix - TARGET_SIZE, y_pix - TARGET_SIZE,),
                      (x_pix + TARGET_SIZE, y_pix + TARGET_SIZE),
                      (1, 255, 1),
                      2)
        cv2.putText(img, str(ind + 1) + "/" + str(DATA_NUMBER_COLLECT), (50, 850),
                    font, 1, (255, 255, 255), 2)
        cv2.imshow('Frame', img)
        cv2.waitKey(1000)

        # Read image
        ret, frame = video_capture.read()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Saving the image
        labels.append([x_nor, y_nor])
        nom = os.path.join(PATH_DATA, series_name + "_" + str(ind).zfill(4) + ".png")
        cv2.imwrite(nom, img_gray)

    # Close webcam and instruction window
    video_capture.release()
    cv2.destroyAllWindows()

    # Save the list of expressions
    labels_filename = os.path.join(PATH_DATA, series_name + "_labels.csv")
    np.savetxt(labels_filename, labels, fmt='%f')

def review_dataset():
    # Load dataset
    dataset = load_dataset()

    # Draw a frame for each dataset entry
    for le, re, lec, rec, lab in zip(*dataset):
        # Initialize
        height, width = 480, 640
        image = np.zeros((height, width), dtype=np.uint8)

        # Draw eyes
        for img, coords in [(le, lec), (re, rec)]:
            w, h = int(coords[2]*width), int(coords[3]*height)
            left, top = int(coords[0]*width - w//2), int(coords[1]*height - h//2)

            img = np.uint8(img*255)
            img = cv2.resize(img, dsize=(w, h))

            image[top: top+h, left: left+w] = img

        # Draw label
        pt = tuple([int(width - lab[0]*width), int(lab[1]*height)])
        cv2.drawMarker(image, pt, 255)

        # Show data
        cv2.imshow("Review dataset", image)
        if cv2.waitKey(0) in [27, ord('q')]:
            break

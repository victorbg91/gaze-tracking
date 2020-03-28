# Imports
import datetime
import glob
import os

from tqdm import tqdm
import numpy as np
import h5py
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


def create_dataset():
    """Compile the data and output a dataset."""
    # Get the name of all the data series.
    series_names = glob.glob(os.path.join(PATH_DATA, "*.csv"))

    # Load the labels.
    print("\nLoading the data")
    labels = []
    for name in series_names:
        label = np.genfromtxt(name, dtype=float)
        print("Loaded {} labels for series {}".format(len(label), name))
        labels.append(label)
    labels = np.concatenate(labels)

    # Load the images
    file_list = [glob.glob(name[:-11] + "_*.png") for name in series_names]
    data = []
    for series in file_list:
        count_img = 0
        for name in series:
            # Load an image in B&W
            img = cv2.imread(name, 0)
            data.append(img)
            count_img += 1

        print("Loaded {} images for series {}".format(count_img, name))

    # Load the Haar cascade
    eye_cascade = cv2.CascadeClassifier(PATH_HAAR_EYES)
    if eye_cascade.empty():
        raise NameError("Couldn't find the Haar Cascade XML file.")

    # Loop information
    data_sz = data[0].shape
    faulty_data = 0

    # Initialize
    data_left_eye = []
    data_right_eye = []
    data_left_eye_coordinates = []
    data_right_eye_coordinates = []
    data_labels = []

    # Extract the eye information
    print("Extracting the eye information")
    for i, img in tqdm(enumerate(data)):
        # Apply the Haar cascade
        eyes = eye_cascade.detectMultiScale(img)

        # Check the data for detection errors
        try:
            # Check if no eyes were detected
            assert type(eyes) is not tuple

            # Check if exactly two eyes were detected
            assert eyes.shape[0] == 2

        # Reject faulty data
        except AssertionError:
            faulty_data += 1
            continue

        # Extract the left and right eye coordinates
        # Keep in mind that the image is naturally inverted in the picture:
        # the left eye will be at the right of the picture
        if eyes[0, 0] < eyes[1, 0]:
            left_eye_coordinates = eyes[1, :]
            right_eye_coordinates = eyes[0, :]
        else:
            left_eye_coordinates = eyes[0, :]
            right_eye_coordinates = eyes[1, :]

        # Extract the eye images
        top, left = left_eye_coordinates[1], left_eye_coordinates[0]
        bottom, right = top + left_eye_coordinates[3], left + left_eye_coordinates[2]
        left_eye = img[top:bottom, left:right]

        top, left = right_eye_coordinates[1], right_eye_coordinates[0]
        bottom, right = top + right_eye_coordinates[3], left + right_eye_coordinates[2]
        right_eye = img[top:bottom, left:right]

        # Resize the eyes
        if DATA_EYES_RESOLUTION is not None and type(DATA_EYES_RESOLUTION) is tuple:
            left_eye = cv2.resize(left_eye, DATA_EYES_RESOLUTION)
            right_eye = cv2.resize(right_eye, DATA_EYES_RESOLUTION)

        # Normalize the image data
        left_eye = np.array(left_eye) / 255.0
        right_eye = np.array(right_eye) / 255.0

        # Normalizing the coordinates
        width = float(data_sz[1])
        height = float(data_sz[0])
        left_eye_coordinates = np.array(left_eye_coordinates, dtype=float)
        right_eye_coordinates = np.array(right_eye_coordinates, dtype=float)
        left_eye_coordinates /= np.array([width, height, width, height])
        right_eye_coordinates /= np.array([width, height, width, height])

        # Change the coordinates to the center of the eye
        left_eye_coordinates[0] += (left_eye_coordinates[2]-left_eye_coordinates[0]) / 2
        left_eye_coordinates[1] += (left_eye_coordinates[3]-left_eye_coordinates[1]) / 2
        right_eye_coordinates[0] += (right_eye_coordinates[2]-right_eye_coordinates[0]) / 2
        right_eye_coordinates[1] += (right_eye_coordinates[3]-right_eye_coordinates[1]) / 2

        # Resize to add the color channel for tensorflow compatibility
        left_eye = left_eye.reshape(DATA_EYES_RESOLUTION[0], DATA_EYES_RESOLUTION[1], 1)
        right_eye = right_eye.reshape(DATA_EYES_RESOLUTION[0], DATA_EYES_RESOLUTION[1], 1)

        # Appending to our data containers
        data_left_eye.append(left_eye)
        data_right_eye.append(right_eye)
        data_left_eye_coordinates.append(left_eye_coordinates)
        data_right_eye_coordinates.append(right_eye_coordinates)
        data_labels.append(labels[i])

    # Print the number of faulty images
    print("There were {} faulty images".format(faulty_data))

    # Convert to numpy arrays for HDF5 compatibility
    data_left_eye = np.array(data_left_eye)
    data_right_eye = np.array(data_right_eye)
    data_left_eye_coordinates = np.array(data_left_eye_coordinates)
    data_right_eye_coordinates = np.array(data_right_eye_coordinates)
    data_labels = np.array(data_labels)

    # Save data to HDF5 file
    with h5py.File(PATH_DATASET, 'w') as f:
        f.create_dataset("data_left_eye", data=data_left_eye)
        f.create_dataset("data_right_eye", data=data_right_eye)
        f.create_dataset("data_left_eye_coordinates", data=data_left_eye_coordinates)
        f.create_dataset("data_right_eye_coordinates", data=data_right_eye_coordinates)
        f.create_dataset("data_labels", data=data_labels)


def load_dataset():
    """
    Load dataset in a training-ready format.

    Returns
    -------
    left_eye : np.ndarray
        Array of shape (n, height, width, 1) containing the left eye images.
    right_eye : numpy.ndarray
        Array of shape (n, height, width, 1) containing the right eye images.
    left_eye_coordinates : numpy.ndarray
        Array of shape (n, 4) containing the coordinates (xc, yc, width, height)
        of the left eye.
    right_eye_coordinates : numpy.ndarray
        Array of shape (n, 4) containing the coordinates (xc, yc, width, height)
        of the right eye.
    labels : numpy.ndarray
        Array of shape (n, 2) containing the coordinates (x, y) of the position
        of the dot on the screen.

    """
    with h5py.File(PATH_DATASET, 'r') as f:
        left_eye = f['data_left_eye'][()]
        right_eye = f['data_right_eye'][()]
        left_eye_coordinates = f['data_left_eye_coordinates'][()]
        right_eye_coordinates = f['data_right_eye_coordinates'][()]
        labels = f['data_labels'][()]

    return left_eye, right_eye, left_eye_coordinates, right_eye_coordinates, labels

# Imports
import datetime
import glob
import os

from tqdm import tqdm
import numpy as np
import cv2

class ImageProcessor:

    """
    Utility class for image processing.

    Attributes
    ----------
    cascade : cv2.CascadeClassifier
        Haar cascade for eye detection.

    """

    # Constants
    PATH_DATA = "./data"
    PATH_CONFIG = "./config"
    PATH_HAAR_EYES = os.path.join(PATH_CONFIG, "haarcascade_eye.xml")
    PATH_INDEX = os.path.join(PATH_DATA, "index.csv")
    PATH_REJECT = os.path.join(PATH_DATA, "reject.csv")

    DATA_EYES_RESOLUTION = (100, 100)
    DATA_NUMBER_COLLECT = 60

    SCREEN_WIDTH = 1600
    SCREEN_HEIGHT = 900
    TARGET_SIZE = 5

    def __init__(self, erase_index=False):
        # Load cascade classifier
        self.cascade = cv2.CascadeClassifier(self.PATH_HAAR_EYES)

        # Erase index if applicable
        if erase_index:
            try:
                os.remove(self.PATH_INDEX)
            except OSError:
                pass
            try:
                os.remove(self.PATH_REJECT)
            except OSError:
                pass

        # Create files if they don't already exist
        if not os.path.exists(self.PATH_INDEX):
            with open(self.PATH_INDEX, "a") as _:
                pass
        if not os.path.exists(self.PATH_REJECT):
            with open(self.PATH_REJECT, "a") as _:
                pass

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

    # TODO modify dataset review
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

    def generate_filelist(self, verbosity=0):
        """Generate a list of all data with their labels"""
        # Get the name of all the data series
        series_names = glob.glob(os.path.join(self.PATH_DATA, "*_labels.csv"))

        # Load the paths
        series_paths = [glob.glob(name[:-11] + "_*.png") for name in series_names]

        # Load the labels
        print("\nLoading the labels")
        series_labels = []
        for name in series_names:
            label = np.genfromtxt(name, dtype=float)
            if verbosity > 0:
                print("Loaded {} labels for series {}".format(len(label), name))
            series_labels.append(label)

        # Assert that we have equal paths and labels for each series
        is_equal = all([len(p) == len(l) for p, l in zip(series_paths, series_labels)])
        is_equal = is_equal and len(series_names) == len(series_paths) == len(series_labels)
        assert is_equal, "The data is misaligned."

        # Assemble the paths and labels
        data = [(p, l) for paths, labels in zip(series_paths, series_labels)
                for p, l in zip(paths, labels)]
        return data

    def _preprocess(self, image):
        """
        Preprocess an image for eye detection.

        Parameters
        ----------
        image : numpy.ndarray
            The input image

        Returns
        -------
        numpy.ndarray
            The output image

        """
        if len(image.shape) != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _detect_eyes(self, image):
        """
        Routine to detect eyes in the image.

        Parameters
        ----------
        image : numpy.ndarray

        Returns
        -------
        eyes : list
            List of eye coordinates with x, y coordinates of the top-left corner and
            width, height of the eyes detected in the image. Units in pixels.

        """
        # Detect the eyes and ensure that 2 eyes are detected
        eyes = self.cascade.detectMultiScale(image)
        if type(eyes) is tuple or len(eyes) != 2:
            return None

        # Identify the left eye
        if eyes[0, 0] < eyes[1, 0]:
            eyes = eyes[::-1, :]

        return eyes

    def index_dataset(self):
        """Index all valid training data and index it for use with dataset generator."""
        # Generate the set of already-indexed files
        indexed = []

        with open(self.PATH_INDEX) as index:
            for line in index:
                path = line.split(",")[0]
                indexed.append(path)
        len_index = len(indexed)

        with open(self.PATH_REJECT) as reject:
            for line in reject:
                indexed.append(line.rstrip("\n"))
        len_reject = len(indexed) - len_index

        print("Currently {} indexed and {} rejected images.".format(
            len_index, len_reject))
        indexed = set(indexed)

        # Generate the data paths
        data = self.generate_filelist()

        # Open index and reject for appending
        index = open(self.PATH_INDEX, "a")
        reject = open(self.PATH_REJECT, "a")

        # Append non-indexed files:
        print("Indexing the files...")
        for path, label in tqdm(data):
            if path in indexed:
                continue

            # Check if image is valid data
            img = cv2.imread(path, 0)
            img = self._preprocess(img)
            res = self._detect_eyes(img)

            # Write to corresponding file
            if res is None:
                reject.write(path + "\n")
            else:
                line = ",".join([path, str(res[0, :]), str(res[1, :]), str(label)]) + "\n"
                index.write(line)

        # Close files
        index.close()
        reject.close()

    def load_index(self):
        """Load all files in the index."""
        paths = []
        lefts = []
        rights = []
        labels = []
        with open(self.PATH_INDEX) as index:
            for line in index:
                line = line.split(",")

                path = line[0]
                left = [int(num) for num in line[1][1:-1].split(" ") if num]
                right = [int(num) for num in line[2][1:-1].split(" ") if num]
                label = [float(num) for num in line[3][1:-2].split(" ") if num]

                paths.append(path)
                lefts.append(left)
                rights.append(right)
                labels.append(label)

        return paths, lefts, rights, labels


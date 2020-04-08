# Imports
import itertools
import os
import glob

import cv2
import numpy as np
import tqdm


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

    def __init__(self):
        # Load cascade classifier
        self.cascade = cv2.CascadeClassifier(self.PATH_HAAR_EYES)

        # Create files if they don't already exist
        if not os.path.exists(self.PATH_INDEX):
            with open(self.PATH_INDEX, "a") as _:
                pass
        if not os.path.exists(self.PATH_REJECT):
            with open(self.PATH_REJECT, "a") as _:
                pass

    def generate_filelist(self):
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
            print("Loaded {} labels for series {}".format(len(label), name))
            series_labels.append(label)

        # Assert that we have equal paths and labels for each series
        is_equal = all([len(p)==len(l) for p, l in zip(series_paths, series_labels)])
        is_equal = is_equal and len(series_names) == len(series_paths) == len(series_labels)
        assert is_equal, "The data is misaligned."

        # Assemble the paths and labels
        data = [[(p, l) for p, l in zip(paths, labels)]
                for paths, labels in zip(series_paths, series_labels)]
        data = list(itertools.chain.from_iterable(data))
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
        eyes = self.cascade.detectMultiScale(image)
        eyes = eyes if len(eyes) == 2 else None
        return eyes

    def _postprocess(self, image, eyes):
        """
        Format the data for compatibility with tensorflow model.

        Parameters
        ----------
        image : numpy.ndarray
            The preprocessed image.
        eyes : list
            List of eye coordinates with x, y coordinates of the top-left corner and
            width, height of the eyes detected in the image. Units in pixels.

        Returns
        -------
        left_eye : numpy.ndarray
            Image of the left eye, resized for compatibility with the model. Colors
            normalized between 0 and 1.
        right_eye : numpy.ndarray
            Image of the right eye, resized for compatibility with the model. Colors
            normalized between 0 and 1.
        left_eye_coordinates : numpy.ndarray
            The (x, y, w, h) coordinates of the left eye. The x, y coordinates are for the
            center of the eye. Units normalized with the width and height of the image.
        right_eye_coordinates : numpy.ndarray
            The (x, y, w, h) coordinates of the left eye. The x, y coordinates are for the
            center of the eye. Units normalized with the width and height of the image.

        """
        # Size of the images
        data_sz = image.shape

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
        left_eye = cv2.resize(left_eye, self.DATA_EYES_RESOLUTION)
        right_eye = cv2.resize(right_eye, self.DATA_EYES_RESOLUTION)

        # Normalize the image data
        left_eye = np.array(left_eye) / 255.0
        right_eye = np.array(right_eye) / 255.0

        # Normalize the coordinates
        width = float(data_sz[1])
        height = float(data_sz[0])
        left_eye_coordinates = np.array(left_eye_coordinates, dtype=float)
        right_eye_coordinates = np.array(right_eye_coordinates, dtype=float)
        left_eye_coordinates /= np.array([width, height, width, height])
        right_eye_coordinates /= np.array([width, height, width, height])

        # Change the coordinates to the center of the eye
        left_eye_coordinates[0] += left_eye_coordinates[2] // 2
        left_eye_coordinates[1] += left_eye_coordinates[3] // 2
        right_eye_coordinates[0] += right_eye_coordinates[2] // 2
        right_eye_coordinates[1] += right_eye_coordinates[3] // 2

        # Resize to add the color channel for tensorflow compatibility
        left_eye = left_eye.reshape(DATA_EYES_RESOLUTION[0], DATA_EYES_RESOLUTION[1], 1)
        right_eye = right_eye.reshape(DATA_EYES_RESOLUTION[0], DATA_EYES_RESOLUTION[1], 1)

        return left_eye, right_eye, left_eye_coordinates, right_eye_coordinates

    def process_image(self, image):
        """Routine to group all eye detection steps."""
        image = self._preprocess(image)
        eyes = self._detect_eyes()
        data = self._postprocess(image, eyes)
        return data

    def index_files(self):
        """Index all valid training data and index it for use with dataset generator."""
        # Generate the set of already-indexed files
        indexed = []

        with open(self.PATH_INDEX) as index:
            for line in index:
                path = line.split(",")[0]
                indexed.append(path)

        with open(self.PATH_REJECT) as reject:
            for line in reject:
                path = line.split(",")[0]
                indexed.append(path)

        indexed = set(indexed)

        # Generate the data paths
        data = self.generate_filelist()

        # Open index and reject for appending
        index = open(self.PATH_INDEX, "a")
        reject = open(self.PATH_REJECT, "a")

        # Append non-indexed files:
        for path, label in tqdm.tqdm(data):
            if path in indexed:
                continue

            # Check if image is valid data
            img = cv2.imread(path, 0)
            img = self._preprocess(img)
            res = self._detect_eyes(img)

            # Append accordingly
            line = path + "," + str(label[0]) + "," + str(label[1]) + "\n"
            if res is None:
                reject.write(line)
            else:
                index.write(line)

        # Close files
        index.close()
        reject.close()

    def load_index(self):
        """Load all files in the index."""
        data = []
        with open(self.PATH_INDEX) as index:
            for line in index:
                line = line.split(",")
                path = line[0]
                label = np.array(line[1:], dtype=float)
                data.append(path, label)

        return data


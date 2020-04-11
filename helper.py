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
        for path, label in tqdm.tqdm(data):
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


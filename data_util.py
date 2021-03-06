# Imports
import datetime
import glob
import os
import functools

import tensorflow as tf
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

    # -------- #
    # RAW DATA #
    # -------- #
    def collect_data(self):
        """Routine to collect new data."""
        # Set the series name
        series_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Checking for conflict with series name
        list_conflict = glob.glob(os.path.join(self.PATH_DATA, series_name + "*.png"))
        if len(list_conflict) > 0:
            answer = input("Conflict detected with this series name.\nDo you want to delete "
                           + "{} items? [y/N]".format(len(list_conflict)))

            # Delete the conflicting files if 'y' or 'Y' is inputed
            if answer == 'y' or answer == 'Y':
                # if True:
                for f in list_conflict:
                    os.remove(f)
                os.remove(os.path.join(self.PATH_DATA, series_name + "_labels.csv"))

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
        img = np.zeros((self.SCREEN_HEIGHT, self.SCREEN_WIDTH), np.uint8)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, 'TRACK THE TARGET', (50, 300), font, 4, (255, 255, 255), 2)
        cv2.imshow('Frame', img)
        cv2.waitKey(1000)

        # Loop to capture frames
        labels = []
        for ind in range(self.DATA_NUMBER_COLLECT):
            # Set the x and y coordinates
            x_nor, y_nor = np.random.rand(), np.random.rand()
            x_pix, y_pix = int(x_nor * self.SCREEN_WIDTH), int(y_nor * self.SCREEN_HEIGHT)

            # Draw the target
            img = np.zeros((self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), np.uint8)
            cv2.rectangle(img,
                          (x_pix - self.TARGET_SIZE, y_pix - self.TARGET_SIZE,),
                          (x_pix + self.TARGET_SIZE, y_pix + self.TARGET_SIZE),
                          (1, 255, 1),
                          2)
            cv2.putText(img, str(ind + 1) + "/" + str(self.DATA_NUMBER_COLLECT), (50, 850),
                        font, 1, (255, 255, 255), 2)
            cv2.imshow('Frame', img)
            cv2.waitKey(1000)

            # Read image
            ret, frame = video_capture.read()
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Saving the image
            labels.append([x_nor, y_nor])
            nom = os.path.join(self.PATH_DATA, series_name + "_" + str(ind).zfill(4) + ".png")
            cv2.imwrite(nom, img_gray)

        # Close webcam and instruction window
        video_capture.release()
        cv2.destroyAllWindows()

        # Save the list of expressions
        labels_filename = os.path.join(self.PATH_DATA, series_name + "_labels.csv")
        np.savetxt(labels_filename, labels, fmt='%f')

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

    # -------- #
    # INDEXING #
    # -------- #
    def _generate_filelist(self, verbosity=0):
        """Generate a list of all data with their labels"""
        # Get the name of all the data series
        series_names = glob.glob(os.path.join(self.PATH_DATA, "*_labels.csv"))

        # Load the paths
        series_paths = [glob.glob(name[:-11] + "_*.png") for name in series_names]

        # Load the labels
        print("Parsing the data folder...")
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

        indexed = set(indexed)

        # Generate the data paths
        data = self._generate_filelist()

        # Open index and reject for appending
        index = open(self.PATH_INDEX, "a")
        reject = open(self.PATH_REJECT, "a")

        # Append non-indexed files:
        print("Before indexing: ", len_index, " indexed and ", len_reject, " rejected images.")
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
                len_reject += 1
            else:
                line = ",".join([path, str(res[0, :]), str(res[1, :]), str(label)]) + "\n"
                index.write(line)
                len_index += 1

        print("After indexing: ", len_index, " indexed and ", len_reject, " rejected images.")

        # Close files
        index.close()
        reject.close()

    def _load_index(self):
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

    # ------- #
    # DATASET #
    # ------- #
    def _parse_eye(self, image, coords, size):
        # Get the coordinates
        left, top, width, height = coords[0], coords[1], coords[2], coords[3]
        img_height, img_width = tf.shape(image)[0], tf.shape(image)[1]

        # Get the eye image
        img = tf.image.crop_to_bounding_box(image, top, left, height, width)
        img = tf.image.resize(img, size)
        img = tf.reshape(img, size + (1,))

        # Cast to float
        left, top = tf.cast(left, tf.float32), tf.cast(top, tf.float32)
        width, height = tf.cast(width, tf.float32), tf.cast(height, tf.float32)
        img_height, img_width = tf.cast(img_height, tf.float32), tf.cast(img_width, tf.float32)

        # Normalize the coordinates
        x = tf.divide(tf.add(left, tf.divide(width, tf.constant(2, tf.float32))), img_width)
        y = tf.divide(tf.add(top, tf.divide(height, tf.constant(2, tf.float32))), img_height)
        w = tf.divide(width, img_width)
        h = tf.divide(height, img_height)
        coord_out = tf.convert_to_tensor([x, y, w, h], dtype=tf.float32)

        return img, coord_out

    def _parse_function(self, path, left_eye_coord, right_eye_coord, label, size):
        # Load the image
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img)
        img = tf.image.convert_image_dtype(img, tf.float32)

        # Get the eye images and normalized coordinates
        left_img, left_coord = self._parse_eye(img, left_eye_coord, size)
        right_img, right_coord = self._parse_eye(img, right_eye_coord, size)

        data = (left_img, right_img, left_coord, right_coord)
        return data, label

    def image_pipeline(self, image_generator, size):
        """Format the inputs to the model for inference."""
        # Initialize
        misfire_counter = 0

        # Start the pipeline
        while True:
            # Grab the next image from the image generator
            frame, eyes = next(image_generator)

            # Format the image
            img = frame / 255.
            img = img.reshape(img.shape + (1,))
            img = tf.image.convert_image_dtype(img, tf.float32)

            # Check if eye coordinates were found
            if eyes is not None:
                # Define the coord tensors
                left_eye_coord = tf.convert_to_tensor(eyes[0, :].astype(int))
                right_eye_coord = tf.convert_to_tensor(eyes[1, :].astype(int))

                # Get the model inputs
                left_img, left_coord = self._parse_eye(img, left_eye_coord, size)
                right_img, right_coord = self._parse_eye(img, right_eye_coord, size)

                # Reshape the tensors for compatibility
                left_img = tf.reshape(left_img, (1,) + left_img.shape)
                right_img = tf.reshape(right_img, (1,) + right_img.shape)
                left_coord = tf.reshape(left_coord, (1,) + left_coord.shape)
                right_coord = tf.reshape(right_coord, (1,) + right_coord.shape)

                inputs = {
                    "input_1": left_img, "input_2": right_img,
                    "input_3": left_coord, "input_4": right_coord}

                # Reset the misfire counter
                misfire_counter = 0

                yield frame, eyes, inputs

            else:
                # Increment the misfire counter
                misfire_counter += 1

            # Stop the program if there were too many consecutive misfires
            if misfire_counter >= 20:
                raise RuntimeError("No data was found for 20 frames")

    def initialize_dataset(self, percent_valid, percent_test, batch, size):
        """
        Initialize a tensorflow dataset pipeline.

        Sources
        -------
        For data pipelines
        https://cs230.stanford.edu/blog/datapipeline/

        # To use python functions
        https://stackoverflow.com/questions/55606909/how-to-use-tensorflow-dataset-with-opencv-preprocessing

        # For buffer size when shuffling the dataset:
        https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle/48096625#48096625

        # To split dataset
        https://stackoverflow.com/questions/51125266/how-do-i-split-tensorflow-datasets/51126863

        # To use TFRECORDS
        https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36

        """
        # Load data
        data = self._load_index()
        num_data = len(data[0])

        # Define parse function
        parsefunc = functools.partial(self._parse_function, size=size)

        # Initialize dataset
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.shuffle(num_data, seed=1, reshuffle_each_iteration=False)
        dataset = dataset.map(parsefunc, num_parallel_calls=4)

        # Split size
        validation_size = int(percent_valid * num_data)
        test_size = int(percent_test * num_data)
        train_size = num_data - validation_size - test_size

        # Split dataset
        data_train = dataset.take(train_size)
        data_valid = dataset.skip(train_size).take(validation_size)
        data_test = dataset.skip(train_size + validation_size)

        # Batch
        data_train = data_train.batch(batch).prefetch(1)
        data_valid = data_valid.batch(batch).prefetch(1)
        data_test = data_test.batch(batch).prefetch(1)

        return data_train, data_valid, data_test

    def review_dataset(self, size):
        # Load dataset
        dataset, _, _ = self.initialize_dataset(0, 0, 1, size)
        iterator = iter(dataset)

        # Draw a frame for each dataset entry
        for data in iterator:
            # Unpack the data
            datapack, label = data
            le, re, lec, rec = datapack

            # Format the data
            le = le.numpy().reshape(size)
            re = re.numpy().reshape(size)
            lec, rec = lec[0], rec[0]

            label = label.numpy()[0]

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
            x = int(width - label[0] * width)
            y = int(label[1] * height)
            pt = (x, y)
            cv2.drawMarker(image, pt, 255)

            # Show data
            cv2.imshow("Review dataset", image)
            if cv2.waitKey(0) in [27, ord('q')]:
                break

    def grabber(self):
        """
        Grab images from the webcam and detect eye positions.

        Returns
        -------
        eyes : numpy.ndarray
            Array containing the coordinates of the detected eyes.

        """
        cap = cv2.VideoCapture(0)

        while True:
            # Grab the frame
            ret, frame = cap.read(0)
            if frame is None:
                break

            # Detect eyes
            frame = self._preprocess(frame)
            eyes = self._detect_eyes(frame)
            yield frame, eyes

        cap.release()

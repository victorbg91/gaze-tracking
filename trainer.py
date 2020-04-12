# Imports
import os
import datetime

import tensorflow as tf

import data_util


class Model:

    """Class for tensorflow model creation, training, and application."""

    # Constants
    PATH_BASE = os.path.abspath(os.getcwd())
    PATH_BASE_LOGS = os.path.join(PATH_BASE, "model", "logs")

    MODEL_PERCENT_VALIDATION = 0.2
    MODEL_PERCENT_TEST = 0.2
    MODEL_IMAGE_SIZE = (64, 64)

    TRAINING_EPOCHS = 1000
    TRAINING_BATCH_SIZE = 32

    def __init__(self):
        self.image_proc = data_util.ImageProcessor()

    # ----- #
    # MODEL #
    # ----- #
    def _define_eye_branch(self):
        """Define the CNN branch for an eye image."""
        input_eye = tf.keras.layers.Input(self.MODEL_IMAGE_SIZE+(1,))

        # eye = tf.keras.layers.Dropout(0.2, input_shape=MODEL_IMAGE_SIZE+(1,)(input_eye)
        eye = input_eye

        eye = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(eye)
        # eye = tf.keras.layers.Dropout(0.2)(eye)
        eye = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(eye)
        # eye = tf.keras.layers.Dropout(0.2)(eye)
        eye = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(eye)

        eye = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(eye)
        # eye = tf.keras.layers.Dropout(0.2)(eye)
        eye = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(eye)
        # eye = tf.keras.layers.Dropout(0.2)(eye)
        eye = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(eye)

        eye = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(eye)
        # eye = tf.keras.layers.Dropout(0.2)(eye)
        eye = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(eye)
        # eye = tf.keras.layers.Dropout(0.2)(eye)
        eye = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(eye)

        # eye = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(eye)
        # # eye = tf.keras.layers.Dropout(0.2)(eye)
        # eye = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(eye)
        # # eye = tf.keras.layers.Dropout(0.2)(eye)
        # eye = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(eye)

        eye = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(eye)  # Flattening
        eye = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(eye)
        flat = tf.keras.layers.Flatten()(eye)

        return input_eye, flat

    def _define_model(self):
        """
        The model for eye-only inputs - the "inspiration" is eye triangulation
        We start by giving separately the left and right eye
        --> the model may detect in which direction the pupil is aimed

        We also want to give the coordinates of the left and right eye
        --> This way, a triangulation may be made

        We also want to give the original width and height
        --> This way, the screen-eye distance may be inferred.
        """
        # Define the eye branches
        input_left_eye, flat_left = self._define_eye_branch()
        input_right_eye, flat_right = self._define_eye_branch()

        # Define the coordinate inputs
        input_left_coord = tf.keras.layers.Input([4])
        input_right_coord = tf.keras.layers.Input([4])

        # Define the main branch
        main = tf.keras.layers.concatenate([flat_left, flat_right, input_left_coord, input_right_coord])
        main = tf.keras.layers.Dense(1000, activation='relu')(main)
        output = tf.keras.layers.Dense(2, activation='linear')(main)

        # Return the model
        model = tf.keras.models.Model(
            inputs=[input_left_eye, input_right_eye, input_left_coord, input_right_coord],
            outputs=[output])
        return model

    def _define_callbacks(self, path_run):
        # Initialize
        callbacks = []

        # Checkpoints
        path_checkpoint = os.path.join(path_run, "checkpoint.ckpt")
        callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=path_checkpoint, monitor='loss', save_best_only=True, verbose=1,
            save_weights_only=True)
        callbacks.append(callback_checkpoint)

        # Logs
        path_logs = os.path.join(path_run, "logfile.csv")
        callback_logger = tf.keras.callbacks.CSVLogger(
            filename=path_logs, separator=',', append=True)
        callbacks.append(callback_logger)

        # Tensorboard
        callback_tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=path_run, histogram_freq=1, write_graph=True, write_images=True)
        callbacks.append(callback_tensorboard)

        return callbacks

    # ---- #
    # DATA #
    # ---- #
    def _parse_eye(self, image, coords):
        # Get the coordinates
        left, top, width, height = coords[0], coords[1], coords[2], coords[3]
        img_height, img_width = tf.shape(image)[0], tf.shape(image)[1]

        # Get the eye image
        img = tf.image.crop_to_bounding_box(image, top, left, height, width)
        img = tf.image.resize(img, self.MODEL_IMAGE_SIZE)
        img = tf.reshape(img, self.MODEL_IMAGE_SIZE + (1,))

        # Normalize the coordinates
        x = (left + width//2) / img_width
        y = (top + height//2) / img_height
        w = width / img_width
        h = height / img_height
        coord_out = tf.convert_to_tensor([x, y, w, h], dtype=float)

        return img, coord_out

    def _parse_function(self, path, left_eye_coord, right_eye_coord, label):
        # Load the image
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img)
        img = tf.image.convert_image_dtype(img, tf.float32)

        # Get the eye images and normalized coordinates
        left_img, left_coord = self._parse_eye(img, left_eye_coord)
        right_img, right_coord = self._parse_eye(img, right_eye_coord)

        data = {"input_1": left_img, "input_2": right_img,
                "input_3": left_coord, "input_4": right_coord}
        return data, label

    def _initialize_dataset(self):
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
        data = self.image_proc.load_index()
        num_data = len(data[0])

        # Initialize dataset
        # tf.random.set_seed(1)
        dataset = tf.data.Dataset.from_tensor_slices(data)
        # dataset = dataset.shuffle(num_data, reshuffle_each_iteration=False)
        dataset = dataset.map(self._parse_function, num_parallel_calls=2)

        # Split size
        validation_size = int(self.MODEL_PERCENT_VALIDATION * num_data)
        test_size = int(self.MODEL_PERCENT_TEST * num_data)
        train_size = num_data - validation_size - test_size

        # Split dataset
        data_train = dataset.take(train_size)
        data_valid = dataset.skip(train_size).take(validation_size)
        data_test = dataset.skip(train_size + validation_size)

        # Batch
        data_train = data_train.batch(self.TRAINING_BATCH_SIZE).prefetch(1)
        data_valid = data_valid.batch(self.TRAINING_BATCH_SIZE).prefetch(1)
        data_test = data_test.batch(self.TRAINING_BATCH_SIZE).prefetch(1)

        return data_train, data_valid, data_test

    # -------- #
    # TRAINING #
    # -------- #
    def train_model(self):
        # Load data
        data_training, data_validation, data_test = self._initialize_dataset()

        # Define the model
        model = self._define_model()
        opt = tf.keras.optimizers.Adam(amsgrad=True)
        model.compile(optimizer=opt, loss='mean_squared_error')
        model.summary()

        # Define callbacks
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path_run = os.path.join(self.PATH_BASE_LOGS, run_id)
        callbacks = self._define_callbacks(path_run)

        # Print the target values
        print(" ")
        print("Training model... ")
        print("We are looking for the following loss values: ")
        print("The 'Better-than-noise' threshold: loss = {:0.2f}".format(1/6))
        print("The 'Better-than-fixed-guess' threshold = {:0.2f}".format(1/12))
        print("For 10% error, we are looking for a loss of {:0.2f}".format(0.01))
        print(" ")

        # Start the model fit
        model.fit(
            data_training, epochs=self.TRAINING_EPOCHS, validation_data=data_validation,
            verbose=1, callbacks=callbacks)

        # Evaluate on the test set
        print("\nResults on the test set")
        model.evaluate(data_test)

        # Save the model
        path_save = os.path.join(path_run, "model.h5")
        model.save(path_save)

# Imports
import os
import datetime
import smtplib
from email.message import EmailMessage
import json

import tensorflow as tf
import numpy as np
from tensorboard.plugins.hparams import api as hp

import data_util


class Model:

    """Class for tensorflow model creation, training, and application."""

    # Paths
    PATH_BASE = os.path.abspath(os.getcwd())
    PATH_BASE_LOGS = os.path.join(PATH_BASE, "logs")
    PATH_EMAIL_CONFIG = os.path.join(PATH_BASE, "config", "email_config.json")
    PATH_MODEL = os.path.join(PATH_BASE, "model")

    # Constants
    DATA_PERCENT_VALIDATION = 0.2
    DATA_PERCENT_TEST = 0.2

    MODEL_BATCH_SIZE = 64
    MODEL_IMAGE_SIZE = (64, 64)
    MODEL_POOL_SIZE = 2
    MODEL_NUM_LAYERS = 2
    MODEL_NUM_UNITS = 2
    MODEL_KERNEL_SIZE = 3

    TRAINING_EPOCHS = 1000
    TRAINING_BATCH_SIZE = 128

    # Hyperparameters
    HP_MAX_TESTS = 50
    HP_LEARNING_RATE = hp.HParam("learning_rate_log", hp.RealInterval(-2., 0.))
    HP_REGULARIZATION = hp.HParam("regularization_log", hp.RealInterval(-7., -0.))
    HP_LEARNING_DECAY = hp.HParam("learning_rate_divisor", hp.Discrete([True, False]))
    HP_LAST_LAYER = hp.HParam("last_layer", hp.Discrete([100, 200, 300, 400, 500]))
    HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam"]))
    HYPERPARAMETERS = [HP_LEARNING_RATE, HP_LEARNING_DECAY, HP_LAST_LAYER, HP_OPTIMIZER, HP_REGULARIZATION]

    def __init__(self):
        self.image_proc = data_util.ImageProcessor()
        self.inferator = None

    def load_model(self):
        loaded = tf.keras.models.load_model(self.PATH_MODEL) # tf.saved_model.load(self.PATH_MODEL)
        loaded.summary()
        inferator = loaded.signatures["serving_default"]
        self.inferator = inferator

    def predict(self, inputs):
        assert self.inferator is not None, "Model was not loaded"
        result = self.inferator(**inputs)["dense_2"].numpy().reshape(-1)
        return result

    def _define_eye_branch_unit(self, inlayer, level, hparams):
        """Define one unit of the eye branch."""
        # Define the convolution layers
        outlayer = inlayer
        filters = 16 * 2 ** level
        regul = 10 ** hparams[self.HP_REGULARIZATION]

        for _ in range(self.MODEL_NUM_LAYERS):
            outlayer = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(self.MODEL_KERNEL_SIZE, self.MODEL_KERNEL_SIZE),
                padding="valid",
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(regul)
            )(outlayer)

        # Finish the unit
        outlayer = tf.keras.layers.MaxPool2D(pool_size=(self.MODEL_POOL_SIZE, self.MODEL_POOL_SIZE))(outlayer)
        return outlayer

    def _define_eye_branch(self, hparams):
        # Define the branch input
        input_eye = tf.keras.layers.Input(self.MODEL_IMAGE_SIZE+(1,))
        outlayer = input_eye

        # Stack the CNN units
        for i in range(self.MODEL_NUM_UNITS):
            outlayer = self._define_eye_branch_unit(outlayer, i, hparams)

        # Finish the eye branch
        outlayer = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(outlayer)
        outlayer = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(outlayer)
        outlayer = tf.keras.layers.Flatten()(outlayer)

        return input_eye, outlayer

    def _define_model(self, hparams):
        """
        Define the model for gaze prediction.

        The "inspiration" is eye triangulation
        We start by giving separately the left and right eye
        --> the model may detect in which direction the pupil is aimed

        We also want to give the coordinates of the left and right eye
        --> This way, a triangulation may be made

        We also want to give the original width and height
        --> This way, the screen-eye distance may be inferred.
        """
        # Define the eye branches
        input_left_eye, flat_left = self._define_eye_branch(hparams)
        input_right_eye, flat_right = self._define_eye_branch(hparams)

        # Define the coordinate inputs
        input_left_coord = tf.keras.layers.Input([4])
        input_right_coord = tf.keras.layers.Input([4])

        # Define the main branch
        last_layer = hparams[self.HP_LAST_LAYER]
        main = tf.keras.layers.concatenate(
            [flat_left, flat_right, input_left_coord, input_right_coord])
        main = tf.keras.layers.Dense(last_layer, activation='relu')(main)
        main = tf.keras.layers.Dense(last_layer, activation='relu')(main)
        output = tf.keras.layers.Dense(2, activation='linear')(main)

        # Return the model
        model = tf.keras.models.Model(
            inputs=[input_left_eye, input_right_eye, input_left_coord, input_right_coord],
            outputs=[output])
        return model

    def _define_callbacks(self, path_run, hparams):
        # Initialize
        callbacks = []

        # Checkpoints
        path_checkpoint = os.path.join(path_run, "checkpoint.ckpt")
        callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=path_checkpoint, monitor='loss', save_best_only=True, verbose=1,
            save_weights_only=False)
        callbacks.append(callback_checkpoint)

        # Early stopping for two conditions:
        # 1 - No improvement after 100 epochs
        # 2 - Loss above 0.08 after 500 epochs
        callback_early_stopping_1 = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=100, mode="min", verbose=2)
        # callback_early_stopping_2 = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=500, baseline=0.08, verbose=2)
        callbacks.append(callback_early_stopping_1)
        # callbacks.append(callback_early_stopping_2)

        # Tensorboard
        callback_tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=path_run, histogram_freq=1, write_graph=True, write_images=True)
        callbacks.append(callback_tensorboard)

        # Hyperparameters
        callback_hyperparameters = hp.KerasCallback(path_run, hparams)
        callbacks.append(callback_hyperparameters)

        return callbacks

    def _send_email(self, results):
        """Send email alert."""

        # Load the email configuration
        with open(self.PATH_EMAIL_CONFIG, "r") as file:
            email_config = json.load(file)

        # Initialize
        msg = EmailMessage()
        msg["Subject"] = "Tests finished"
        msg["From"] = email_config["from"]
        msg["To"] = email_config["to"]

        # Set the message header
        text = "A new batch of tests is finished."

        text += "\n\nAs a reminder, we are looking for the following MSE values: "
        text += "\n'Better-than-noise' threshold = {:0.2f}".format(1/6)
        text += "\n'Better-than-fixed-guess' threshold = {:0.2f}".format(1/12)
        text += "\nOur objective of 10% error threshold = {:0.2f}".format(0.01)

        # Set the email info
        keys = sorted(results.keys())[:10]
        top_params = [results[key] for key in keys]
        top_params = [{param.name: params[param] for param in params}
                      for params in top_params]

        text += "\n\n" + "-"*50
        for key, params in zip(keys, top_params):
            text += "\n\n" + str(round(key, 2))
            text += "\n" + str(params)

        # Add the text to the email
        msg.set_content(text)

        # Send the email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.ehlo()
            server.login(email_config["from"], email_config["password"])
            server.ehlo()

            server.send_message(msg)

    def train_model(self, hparams):
        # Load data
        data_training, data_validation, data_test = self.image_proc.initialize_dataset(
            self.DATA_PERCENT_VALIDATION, self.DATA_PERCENT_TEST,
            self.TRAINING_BATCH_SIZE, self.MODEL_IMAGE_SIZE)

        # Define the model
        try:
            model = self._define_model(hparams)
            model.summary()
        except ValueError as err:
            print("Could not define the model - possibly incompatible hyperparameters.")
            print("ERROR: ", err)
            return np.inf

        # Define the optimizer
        if hparams[self.HP_OPTIMIZER] == "adam":
            opt = tf.keras.optimizers.Adam()
        elif hparams[self.HP_OPTIMIZER] is "sgd":
            learning_rate = 10 ** hparams[self.HP_LEARNING_RATE]
            learning_decay = learning_rate/self.TRAINING_EPOCHS if hparams[self.HP_LEARNING_DECAY] else 0.
            opt = tf.keras.optimizers.SGD(lr=learning_rate, decay=learning_decay)

        # Define the model
        model.compile(optimizer=opt, loss='mean_squared_error')

        # Define callbacks
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path_run = os.path.join(self.PATH_BASE_LOGS, run_id)
        callbacks = self._define_callbacks(path_run, hparams)

        # Start the model fit
        try:
            model.fit(
                data_training, epochs=self.TRAINING_EPOCHS, validation_data=data_validation,
                verbose=0, callbacks=callbacks)
        except tf.errors.ResourceExhaustedError as err:
            print("Error when fitting the model, ran out of memory.")
            print(err)
            return np.inf
        except tf.errors.InvalidArgumentError as err:
            print("Model had a NaN, possibly explosive gradient problem")
            print("After inspection, this problem was cause by improper normalization of the inputs.")
            print(err)
            return np.inf

        # Evaluate on the test set
        print("\nResults on the test set")
        result = model.evaluate(data_test)

        return result

    def launch_training_batch(self):
        # Launch the tests
        results = {}
        for _ in range(self.HP_MAX_TESTS):
            # Clear the previous session
            tf.keras.backend.clear_session()

            # Pick hyperparameters at random
            hparams = {param: param.domain.sample_uniform() for param in self.HYPERPARAMETERS}

            # Print a run message
            msg = {param.name: hparams[param] for param in hparams}
            print("\nStarting a new model with parameters:\n", msg)

            # Train the model and append the results
            res = self.train_model(hparams)
            results[res] = hparams

        # Send alert email
        self._send_email(results)

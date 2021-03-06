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


class EyeBranch(tf.keras.layers.Layer):

    """

    Custom block for eye images.

    """

    def __init__(self, regularizer):
        # Initialize
        super(EyeBranch, self).__init__()

        # BLOCK A
        self.A1_conv = tf.keras.layers.Conv2D(
            filters=16, kernel_size=(3, 3),
            padding="valid", activation='relu',
            name="A1_conv",
            kernel_regularizer=regularizer
        )
        self.A2_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="A2_pool")

        # BLOCK B
        self.B1_conv = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="valid",
            activation='relu',
            name="B1_conv",
            kernel_regularizer=regularizer
        )
        self.B2_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="B2_pool")

        # BLOCK C
        self.C1_conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            activation='relu',
            name="C1_conv",
            kernel_regularizer=regularizer
        )
        self.C2_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="C2_pool")
        self.C3_flat = tf.keras.layers.Flatten(name="C3_flat")

    def __call__(self, inputs):
        x = self.A1_conv(inputs)
        x = self.A2_pool(x)

        x = self.B1_conv(x)
        x = self.B2_pool(x)

        x = self.C1_conv(x)
        x = self.C2_pool(x)
        x = self.C3_flat(x)

        return x


class VarianceRegularizer(tf.keras.regularizers.Regularizer):

    """

    Custom kernel regularization to penalize low weight variance.

    """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x):
        return self.alpha * tf.math.reciprocal(tf.math.reduce_variance(x))

    def get_config(self):
        return {"alpha": float(self.alpha)}


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

    MODEL_IMAGE_SIZE = (64, 64)
    MODEL_POOL_SIZE = 2
    MODEL_NUM_LAYERS = 1
    MODEL_NUM_UNITS = 2
    MODEL_KERNEL_SIZE = 3

    TRAINING_EPOCHS = 1000
    TRAINING_BATCH_SIZE = 64

    # Hyperparameters
    HP_MAX_TESTS = 50
    HP_LEARNING_RATE = hp.HParam("learning_rate_log", hp.RealInterval(-4., -3.))
    HP_VAR_REGULARIZATION = hp.HParam("var_regul_log", hp.RealInterval(-10., -0.))
    HP_LEARNING_DECAY = hp.HParam("learning_rate_divisor", hp.Discrete([True]))
    HP_LAST_LAYER = hp.HParam("last_layer", hp.Discrete([200, 300, 400]))
    HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam"]))

    HYPERPARAMETERS = [HP_LEARNING_RATE, HP_LEARNING_DECAY, HP_LAST_LAYER, HP_OPTIMIZER, HP_VAR_REGULARIZATION]

    def __init__(self):
        self.image_proc = data_util.ImageProcessor()
        self.inferator = None

    def load_model(self):
        loaded = tf.keras.models.load_model(self.PATH_MODEL)  # tf.saved_model.load(self.PATH_MODEL)
        loaded.summary()
        inferator = loaded.signatures["serving_default"]
        self.inferator = inferator

    def predict(self, inputs):
        assert self.inferator is not None, "Model was not loaded"
        result = self.inferator(**inputs)["dense_2"].numpy().reshape(-1)
        return result

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
        # Initialize
        var_regularization = VarianceRegularizer(10 ** hparams[self.HP_VAR_REGULARIZATION])
        last_layer = hparams[self.HP_LAST_LAYER]

        # Input layers
        input_left_eye = tf.keras.layers.Input(self.MODEL_IMAGE_SIZE + (1,))
        input_right_eye = tf.keras.layers.Input(self.MODEL_IMAGE_SIZE + (1,))
        input_left_coord = tf.keras.layers.Input([4])
        input_right_coord = tf.keras.layers.Input([4])

        # Define the weight-sharing eye branch
        eye_branch = EyeBranch(var_regularization)
        flat_left = eye_branch(input_left_eye)
        flat_right = eye_branch(input_right_eye)

        # Define the main branch
        main = tf.keras.layers.concatenate(
            [flat_left, flat_right, input_left_coord, input_right_coord], name="D1_concat")
        main = tf.keras.layers.Dense(
            last_layer, activation='relu', name="D2_dense",
            kernel_regularizer=var_regularization)(main)
        main = tf.keras.layers.Dense(
            last_layer, activation='relu', name="D3_dense",
            kernel_regularizer=var_regularization)(main)
        output = tf.keras.layers.Dense(2, name="D4_output", activation='sigmoid')(main)

        # Return the model
        model = tf.keras.models.Model(
            inputs=[input_left_eye, input_right_eye, input_left_coord, input_right_coord],
            outputs=[output])
        return model

    @staticmethod
    def _define_callbacks(path_run, hparams):
        # Initialize
        callbacks = []
        monitor = "loss"

        # Checkpoints
        path_checkpoint = os.path.join(path_run, "checkpoint")
        callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=path_checkpoint, monitor=monitor,
            save_best_only=True, verbose=1,
            save_weights_only=False)
        callbacks.append(callback_checkpoint)

        # Early stopping if there is no improvement over 100 epochs
        callback_early_stopping_1 = tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=100, mode="min", verbose=2)
        callbacks.append(callback_early_stopping_1)

        # Tensorboard
        callback_tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=path_run, update_freq="epoch", histogram_freq=10,
            write_graph=True, write_images=False,
        )
        callbacks.append(callback_tensorboard)

        # Hyperparameters
        callback_hyperparameters = hp.KerasCallback(path_run, hparams)
        callbacks.append(callback_hyperparameters)

        # TODO implement callback to put images in tensorboard
        # Source:
        # https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras?rq=1

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

    def train_model(self, hparams, load_weights=False):
        # Define run options
        run_options = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)

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

        # Define the learning parameters
        learning_rate = 10 ** hparams[self.HP_LEARNING_RATE]
        learning_decay = learning_rate/self.TRAINING_EPOCHS if hparams[self.HP_LEARNING_DECAY] else 0.

        # Define the optimizer
        if hparams[self.HP_OPTIMIZER] == "adam":
            opt = tf.keras.optimizers.Adam(lr=learning_rate, decay=learning_decay)
        elif hparams[self.HP_OPTIMIZER] is "sgd":
            opt = tf.keras.optimizers.SGD(lr=learning_rate, decay=learning_decay)
        else:
            raise ValueError("Unknown optimizer, expected 'adam' or 'sgd'")

        # Define the model
        model.compile(
            optimizer=opt, loss='mean_squared_error', metrics=["mean_squared_error"], options=run_options)

        if load_weights:
            try:
                # Load the weights and create a dictionary of weights
                imported = tf.saved_model.load(self.PATH_MODEL)
                weights = {layer.name: layer.numpy() for layer in imported.variables}
            except OSError:
                weights = {}
                print("WARNING: COULD NOT LOAD MODEL, starting from standard weight initialization")

            # Initialize the model with the loaded weights
            for i, layer in enumerate(model.layers):
                try:
                    # Get the weight for the layer
                    name = layer.name
                    kernel_weights = weights.get(name + "/kernel:0")
                    bias_weights = weights.get(name + "/bias:0")
                    if kernel_weights is None or bias_weights is None:
                        init_weights = []
                    else:
                        init_weights = [kernel_weights, bias_weights]

                    # Set the weights
                    layer.set_weights(init_weights)

                except ValueError:
                    print("Could not load the weights in the layer: ", layer.name)

        # Define callbacks
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path_run = os.path.join(self.PATH_BASE_LOGS, run_id)
        callbacks = self._define_callbacks(path_run, hparams)

        # Start the model fit
        try:
            model.fit(
                data_training, epochs=self.TRAINING_EPOCHS, validation_data=data_validation,
                verbose=2, callbacks=callbacks)
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

    def launch_training_batch(self, load_weights=False):
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
            res = self.train_model(hparams, load_weights)
            if type(res) is list:
                res = res[-1]
            results[res] = hparams

        # Send alert email
        self._send_email(results)

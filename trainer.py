# Imports
import os
import datetime

import tensorflow as tf
import numpy as np

# Constants
PATH_BASE = os.path.abspath(os.getcwd())
PATH_BASE_LOGS = os.path.join(PATH_BASE, "model", "logs")

MODEL_PERCENT_VALIDATION = 0.2
MODEL_PERCENT_TEST = 0.2
MODEL_IMAGE_SIZE = (100, 100, 1)
MODEL_TRAINING_EPOCHS = 200


def _define_eye_branch():
    """Define the CNN branch for an eye image."""
    input_eye = tf.keras.layers.Input(MODEL_IMAGE_SIZE)

    # eye = tf.keras.layers.Dropout(0.2, input_shape=eye_input_sz)(input_eye)
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


def _define_model():
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
    input_left_eye, flat_left = _define_eye_branch()
    input_right_eye, flat_right = _define_eye_branch()

    # Define the coordinate inputs
    input_left_coord = tf.keras.layers.Input([4], name='Input_left_coord')
    input_right_coord = tf.keras.layers.Input([4], name='Input_right_coord')

    # Define the main branch
    main = tf.keras.layers.concatenate([flat_left, flat_right, input_left_coord, input_right_coord])
    main = tf.keras.layers.Dense(1000, activation='relu')(main)
    output = tf.keras.layers.Dense(2, activation='linear')(main)

    # Return the model
    model = tf.keras.models.Model(
        inputs=[input_left_eye, input_right_eye, input_left_coord, input_right_coord],
        outputs=[output])
    return model


def _split_data(data):
    # Define indices for shuffling
    num_data = data[0].shape[0]
    ind_shuffle = np.arange(num_data)
    np.random.shuffle(ind_shuffle)

    # Define number of validation and test data entries
    num_val = int(MODEL_PERCENT_VALIDATION * num_data)
    num_test = int(MODEL_PERCENT_TEST * num_data)

    # Define validation, test, and training indices
    shuffle_indices_validation = ind_shuffle[:num_val]
    shuffle_indices_test = ind_shuffle[num_val:num_val+num_test]
    shuffle_indices_training = ind_shuffle[num_val+num_test:]

    # Separate data
    data_validation = [d[shuffle_indices_validation, ...] for d in data]
    label_validation = data_validation.pop()
    data_test = [d[shuffle_indices_test, ...] for d in data]
    label_test = data_test.pop()
    data_training = [d[shuffle_indices_training, ...] for d in data]
    label_training = data_training.pop()
    return data_training, label_training, data_validation, label_validation, data_test, label_test


def _define_callbacks(path_run):
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


def train_model(data):
    # Load data
    loaded_data = _split_data(data)
    data_training, label_training = loaded_data[:2]
    data_validation, label_validation = loaded_data[2:4]
    data_test, label_test = loaded_data[4:]

    # Define the model
    model = _define_model()
    opt = tf.keras.optimizers.Adam(amsgrad=True)
    model.compile(optimizer=opt, loss='mean_squared_error')
    model.summary()

    # Define callbacks
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path_run = os.path.join(PATH_BASE_LOGS, run_id)
    callbacks = _define_callbacks(path_run)

    # Print the target values
    print(" ")
    print("Training model... ")
    print("We are looking for the following loss values: ")
    print("The 'Better-than-noise' threshold: loss = {:0.2f}".format(1/6))
    print("The 'Better-than-fixed-guess' threshold = {:0.2f}".format(1/12))
    print("For 10% error, we are looking for a loss of {:0.2f}".format(0.01))
    print(" ")

    # Start the model fit
    model.fit(data_training, label_training, epochs=MODEL_TRAINING_EPOCHS,
              validation_data=(data_validation, label_validation),
              verbose=2, callbacks=callbacks)

    # Evaluate on the test set
    print("\nResults on the test set")
    model.evaluate(data_test, label_test)

    # Save the model
    path_save = os.path.join(path_run, "model.h5")
    model.save(path_save)

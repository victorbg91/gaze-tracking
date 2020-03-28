# Imports
import os

import tensorflow as tf
import numpy as np

# Constants
MODEL_PERCENT_VALIDATION = 0.2
MODEL_PERCENT_TEST = 0.2
MODEL_IMAGE_SIZE = (100, 100)


def model_conv(input_sz):
    model = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.2, input_shape=input_sz),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation=tf.nn.relu),#76x56
        # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),#38x28
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation='linear')
    ])

    return model


# The model for eye-only inputs - the "inspiration" is eye triangulation
# We start by giving separately the left and right eye
# --> the model may detect in which direction the pupil is aimed
#
# We also want to give the coordinates of the left and right eye
# --> This way, a triangulation may be made
#
# We also want to give the original width and height
# --> This way, the screen-eye distance may be inferred.
def model_conv_eyes(eye_input_sz):
    # INPUTS
    input_lefteye = tf.keras.layers.Input(eye_input_sz, name='Input_left_eye')
    input_righteye = tf.keras.layers.Input(eye_input_sz, name='Input_right_eye')
    input_leftcoord = tf.keras.layers.Input([4], name='Input_left_coord')
    input_rightcoord = tf.keras.layers.Input([4], name='Input_right_coord')

    # LEFT EYE BRANCH
    # left = tf.keras.layers.Dropout(0.2, input_shape=eye_input_sz)(input_lefteye)
    left = input_lefteye

    left = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(left)
    # left = tf.keras.layers.Dropout(0.2)(left)
    left = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(left)
    # left = tf.keras.layers.Dropout(0.2)(left)
    left = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(left)

    left = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(left)
    # left = tf.keras.layers.Dropout(0.2)(left)
    left = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(left)
    # left = tf.keras.layers.Dropout(0.2)(left)
    left = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(left)

    left = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(left)
    # left = tf.keras.layers.Dropout(0.2)(left)
    left = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(left)
    # left = tf.keras.layers.Dropout(0.2)(left)
    left = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(left)

    # left = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(left)
    # # left = tf.keras.layers.Dropout(0.2)(left)
    # left = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(left)
    # # left = tf.keras.layers.Dropout(0.2)(left)
    # left = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(left)

    left = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(left)  # Flattening
    left = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(left)
    flat_left = tf.keras.layers.Flatten(name="flat_left")(left)

    # RIGHT EYE BRANCH
    # right = tf.keras.layers.Dropout(0.2)(input_righteye)
    right = input_righteye

    right = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', name='RA1')(right)
    # right = tf.keras.layers.Dropout(0.2)(right)
    right = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', name='RA2')(right)
    # right = tf.keras.layers.Dropout(0.2)(right)
    right = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='RA3')(right)

    right = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', name='RB1')(right)
    # right = tf.keras.layers.Dropout(0.2)(right)
    right = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', name='RB2')(right)
    # right = tf.keras.layers.Dropout(0.2)(right)
    right = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='RB3')(right)

    right = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='RC1')(right)
    # right = tf.keras.layers.Dropout(0.2)(right)
    right = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='RC2')(right)
    # right = tf.keras.layers.Dropout(0.2)(right)
    right = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='RC3')(right)

    # right = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', name='RD1')(right)
    # # right = tf.keras.layers.Dropout(0.2)(right)
    # right = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', name='RD2')(right)
    # # right = tf.keras.layers.Dropout(0.2)(right)
    # right = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='RD3')(right)

    right = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(right)
    right = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(right)
    flat_right = tf.keras.layers.Flatten(name="flat_right")(right)

    # Adding all inputs together
    main = tf.keras.layers.concatenate([flat_left, flat_right, input_leftcoord, input_rightcoord])

    # Main branch
    main = tf.keras.layers.Dense(1000, activation='relu')(main)

    # Output
    output = tf.keras.layers.Dense(2, activation='linear')(main)

    # Returning the model
    model = tf.keras.models.Model(inputs=[input_lefteye,
                                          input_righteye,
                                          input_leftcoord,
                                          input_rightcoord],
                                  outputs=[output])
    return model


def train_network(data, checkpointfolder=None, outpath=None, logpath=None, tfboardpath=None, perc_val=0.2, perc_test=0.2):
    print("\nTraining the network")
    # ---------------- #
    # DATA PREPARATION #
    # ---------------- #
    # Indices for shuffling
    num_data = data[0].shape[0]
    ind_shuffle = np.arange(num_data)
    np.random.shuffle(ind_shuffle)

    # Number of validation and test data entries
    num_val = int(perc_val*num_data)
    num_test = int(perc_test*num_data)

    # Validation, test, and training data points
    shuf_va = ind_shuffle[:num_val]
    shuf_te = ind_shuffle[num_val : num_val+num_test]
    shuf_tr = ind_shuffle[num_val+num_test:]

    # Validation data
    data_validation = [data[0][shuf_va, :, :, :],
                       data[1][shuf_va, :, :, :],
                       data[2][shuf_va, :],
                       data[3][shuf_va, :]]
    label_validation = data[4][shuf_va, :]

    # Test data
    data_test = [data[0][shuf_te, :, :, :],
                 data[1][shuf_te, :, :, :],
                 data[2][shuf_te, :],
                 data[3][shuf_te, :]]
    label_test = data[4][shuf_te, :]

    # Training data
    data_training = [data[0][shuf_tr, :, :, :],
                     data[1][shuf_tr, :, :, :],
                     data[2][shuf_tr, :],
                     data[3][shuf_tr, :]]
    label_training = data[4][shuf_tr, :]

    # ----------------- #
    # MODEL PREPARATION #
    # ----------------- #
    # Defining our model
    eye_input_sz = data[0].shape[1:]
    model = model_conv_eyes(eye_input_sz)

    # Compiling
    # model.compile(optimizer='adam',
    #               loss='mean_squared_error',
    #               )

    # The optimizer
    opt = tf.keras.optimizers.Adam(amsgrad=True)
    # opt = tf.keras.optimizers.SGD(learning_rate=0.005,
    #                         momentum=0.1,
    #                         name="SGD")

    # Compiling
    model.compile(optimizer=opt,
                  loss='mean_squared_error')

    # Loading pre-existing weights
    # if checkpointfolder is not None:
    #     print("ASDASDSAD")
    #     latest = tf.train.latest_checkpoint(checkpointfolder)
    #     print(latest)
    #
    #     # # We load the proposed model
    #     model.load_weights(latest)
    #     # model.assert_consumed()
    #     model.evaluate(data_test, label_test)
    #     print("Loaded the inputted weights.")
    #
    #     try:
    #         pass
    #         # We load the proposed model
    #         # model.load_weights(latest)
    #         # model.assert_consumed()
    #         # model.evaluate(data_test, label_test)
    #         # print("Loaded the inputted weights.")
    #     except:
    #         print("ERROR: Could not load the inputted weights.")

    # Printing the summary of our model
    model.summary()

    # --------- #
    # CALLBACKS #
    # --------- #
    # List of callbacks
    cbs = []

    # Checkpoints
    if checkpointfolder is not None:
        cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpointfolder, "checkpoint.ckpt"),
            monitor='loss',
            save_best_only=True,
            verbose=1,
            save_weights_only=True,
        )
        cbs.append(cb_checkpoint)

    # Logfiles
    if logpath is not None:
        cb_logger = tf.keras.callbacks.CSVLogger(
            filename=logpath,
            separator=',',
            append=True,
        )
        cbs.append(cb_logger)

    # Tensorboard
    if tfboardpath is not None:
        cb_tfboard = tf.keras.callbacks.TensorBoard(
            log_dir=tfboardpath,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
        )
        cbs.append(cb_tfboard)

    # -------------- #
    # MODEL TRAINING #
    # -------------- #
    # Starting the model fit
    model.fit(data_training,
              label_training,
              epochs=5000,
              validation_data=(data_validation, label_validation),
              verbose=2,
              callbacks=cbs
              )

    # Evaluating the test set
    print("\nResults on the test set")
    model.evaluate(data_test, label_test)

    # Print the "Better than noise value"
    print("\nThe 'Better-than-noise' threshold: loss = {:0.2f}".format(1 / 6))
    print("\nThe 'Better-than-fixed-guess threshold = {:0.2f}'".format(1/12))
    print("For 10% error, we are looking for a loss of {:0.2f}".format(0.01))

    # Saving the model
    model.save(outpath)


def save_complete_model(model, inpath, outpath):
    model.load_weights(inpath)
    tf.saved_model.save(model, outpath)

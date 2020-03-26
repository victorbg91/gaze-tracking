import argparse
import os

import tensorflow as tf
import numpy as np
import h5py
import cv2

import data

# ---------------- #
# LOADING DATA SET #
# ---------------- #
# Loading the HDF5 dataset
def loadHDF5(inpath):
    with h5py.File(inpath, 'r') as f:
        dataLeftEye = f['dataLeftEye'][()]
        dataRightEye = f['dataRightEye'][()]
        dataLeftCoord = f['dataLeftCoord'][()]
        dataRightCoord = f['dataRightCoord'][()]
        dataLabels = f['dataLabels'][()]

    return dataLeftEye, dataRightEye, dataLeftCoord, dataRightCoord, dataLabels

# -------------------- #
# TRAIN NEURAL NETWORK #
# -------------------- #
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


# Routine to train the neural network
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


# quick routine to validate my calculus for the better-than-noise threshold
def monte_carlo(iter_num):
    """

    :param iter_num: Number of iterations for the Monte Carlo simulation.
    :return: mse
    """
    v = np.random.random((iter_num, 2))
    d1 = v[:, 0] - v[:, 1]  # random guess
    d2 = v[:, 0] - 0.5  # fixed guess of 0.5
    s = d2 ** 2
    mse = np.mean(s)
    return mse

def save_complete_model(model, inpath, outpath):
    model.load_weights(inpath)
    tf.saved_model.save(model, outpath)

# ------------------ #
# WEBCAM APPLICATION #
# ------------------ #
# Image treatment
def acquire_image():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise Exception("Could not open webcam")

    # Capturing image from webcam and running smile detection
    while True:
        # Read image
        ret, frame = video_capture.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yield img

def extract_eye(gen_img, res_out=(100,100)):
    # Loading the Haar cascade
    eye_cascade = cv2.CascadeClassifier('./config/haarcascade_eye.xml')
    if eye_cascade.empty():
        raise NameError("Couldn't find the Haar Cascade XML file.")

    # Loop information
    # data_sz = data[0].shape

    # dataLeftEye = []
    # dataRightEye = []
    # dataLeftCoord = []
    # dataRightCoord = []
    # dataLabels = []

    # Extracting the eye information
    while True:
        img = next(gen_img)
        data_sz = img.shape
        eyes = eye_cascade.detectMultiScale(img)

        # Making sure we detect exactly 2 eyes
        try:
            assert (eyes.shape[0] == 2)
        except AssertionError:
            continue

        # Extracting the left and right eye coordinates
        # Keep in mind that the image is naturally inverted: the left eye will be at the right of the picture
        # They are ordered as [x, y, width, height]
        # It is given in pixel values
        if eyes[0, 0] < eyes[1, 0]:
            lec = eyes[1, :]
            rec = eyes[0, :]
        else:
            lec = eyes[0, :]
            rec = eyes[1, :]

        # Extracting the eyes
        le = img[lec[1]:lec[1] + lec[3], lec[0]:lec[0] + lec[2]]
        re = img[rec[1]:rec[1] + rec[3], rec[0]:rec[0] + rec[2]]

        # Resizing the eyes
        if res_out is not None and type(res_out) is tuple:
            le = cv2.resize(le, res_out)
            re = cv2.resize(re, res_out)

        # Normalizing the image data
        le = np.array(le) / 255.0
        re = np.array(re) / 255.0

        # Normalizing the coordinates
        w = float(data_sz[1])
        h = float(data_sz[0])
        lec = np.array(lec, dtype=float)
        rec = np.array(rec, dtype=float)
        lec[0] /= w
        lec[2] /= w
        rec[0] /= w
        rec[2] /= w
        lec[1] /= h
        lec[3] /= h
        rec[1] /= h
        rec[3] /= h

        # Changing our coordinates to the center of the eye
        lec[0] += lec[2] / 2
        lec[1] += lec[3] / 2
        rec[0] += rec[2] / 2
        rec[1] += rec[3] / 2

        # Resizing to add the color channel for compatibility with tensorflow convolution layers
        le = le.reshape(res_out[0], res_out[1], 1)
        re = re.reshape(res_out[0], res_out[1], 1)

        # Returning the extracted info
        yield le, re, lec, rec

def webcam_gaze(modelpath, eye_gen):
    # Loading the neural network
    model = tf.saved_model.load(export_dir=modelpath, tags=None)

    # Opening the OpenCV window
    # The resolution of the screen
    scr_width = 1600
    scr_height = 900

    # The half-size of the target
    targ_sz = 5

    # Opening display window
    cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Display the gaze text and wait 1 second.
    img = np.zeros((scr_height, scr_width), np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, 'BIG BROTHER IS WATCHING', (50, 300), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Frame', img)
    cv2.waitKey(1000)

    # Loop to capture frames
    while True:
        # Getting the eyes information
        eyes = next(eye_gen)

        # Running the model
        xp, yp = model.evaluate(eyes)

        # Draw the detected location
        img = np.zeros((scr_height, scr_width, 3), np.uint8)
        cv2.rectangle(img,
                      (xp - targ_sz, yp - targ_sz,),
                      (xp + targ_sz, yp + targ_sz),
                      (255, 255, 255),
                      2)

        # Display instructions to exit
        cv2.putText(frame, "Press 'ESC' to exit", (25, 450), font, 1, (255, 255, 255), 1)

        # Showing the image
        cv2.imshow("Face detection", frame)

        # Catching the escape key
        key_press = cv2.waitKey(16) & 0xFF
        if key_press == 27:
            break

    # Closing the window and webcam
    video_capture.release()
    cv2.destroyAllWindows()


def parse_inputs():
    """
    Parse the input arguments for actions tu run.

    :return: Namespace containing the actions to run.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collect-data", action="store_true", help="Collect new data")
    parser.add_argument("-d", "--create-dataset", action="store_true", help="Create a new dataset")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse inputs
    args = parse_inputs()

    # Check args to collect data
    if args.collect_data:
        data.collect_data()

    if args.create_dataset:
        data.create_dataset()

    # Loading the data and training the model
    # data = loadHDF5("./data/compilation.h5")
    # train_network(data,
    #               checkpointfolder="./model/",
    #               outpath="./model/model.h5",
    #               logpath='./model/logfile.csv',
    #               tfboardpath=None,#"./model/logboard/",
    #               perc_val=0.2,
    #               perc_test=0.2,
    #               )

    # Loading a checkpoint and saving the corresponding model.
    # model = model_conv_eyes((100, 100, 1))
    # save_complete_model(model=model,
    #                     inpath="./model/checkpoint.ckpt",
    #                     outpath="./model/savedmodels/1/")

    # img_gen = acquire_image()
    # eye_gen = extract_eye(img_gen, res_out=(100, 100))
    # webcam_gaze(modelpath="./model/savedmodels/1/", eye_gen=eye_gen)


    # Runnning the webcam application
    # webcam_smile(inpath="./model/model.h5", res_img=(64, 64))

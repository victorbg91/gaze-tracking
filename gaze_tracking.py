"""
Author: Victor BG
Date: November 15, 2019

This script intends to train a gaze tracking network

"""

# ------- #
# IMPORTS #
# ------- #
# from win32api import GetSystemMetrics
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import datetime
import glob
import sys
import cv2
import os
import h5py


# ----------------- #
# CREATING DATA SET #
# ----------------- #
def create_data(num_expression, series_name, folder="./data/"):
    # The resolution of the screen
    scr_width = 1600
    scr_height = 900

    # The half-size of the target
    targ_sz = 5

    # Checking for conflict with series name
    list_conflict = glob.glob(folder + series_name + "*.png")
    if len(list_conflict) > 0:
        answer = input("Conflict detected with this series name.\nDo you want to delete " \
                       + "{} items? [y/N]".format(len(list_conflict)))

        # Delete the conflicting files if 'y' or 'Y' is inputed
        if answer == 'y' or answer == 'Y':
            # if True:
            for f in list_conflict:
                os.remove(f)
            os.remove(folder + series_name + "_labels.csv")

        # Abort the program otherwise.
        else:
            print("Acquisition aborted")
            return -1

    # Label container
    expression = []

    # Opening video capture
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise Exception("Could not open webcam")

    # Opening display window
    # print(GetSystemMetrics(0))
    cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Display the gaze instructions and wait 1 second.
    img = np.zeros((scr_height, scr_width), np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, 'TRACK THE TARGET', (50, 300), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Frame', img)
    cv2.waitKey(1000)

    # Loop to capture frames
    for ind in range(num_expression):
        # Set the x and y coordinates
        x_nor, y_nor = np.random.rand(), np.random.rand()
        x_pix, y_pix = int(x_nor * scr_width), int(y_nor * scr_height)

        # Draw the target
        img = np.zeros((scr_height, scr_width, 3), np.uint8)
        cv2.rectangle(img,
                      (x_pix - targ_sz, y_pix - targ_sz,),
                      (x_pix + targ_sz, y_pix + targ_sz),
                      (1, 255, 1),
                      2)

        cv2.putText(img, str(ind + 1) + "/" + str(num_expression), (50, 850), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Frame', img)
        cv2.waitKey(1000)

        # Read image
        ret, frame = video_capture.read()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Saving the image
        try:
            # Save image and label
            expression.append([x_nor, y_nor])
            nom = folder + series_name + "_" + str(ind).zfill(4) + ".png"
            cv2.imwrite(nom, img_gray)
        except:
            print("Could not save the image.")

    # Close webcam and instruction window
    video_capture.release()
    cv2.destroyAllWindows()

    # Save the list of expressions
    nom_fiche = folder + series_name + "_labels.csv"
    np.savetxt(nom_fiche, expression, fmt='%f')


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


# Formatting the dataset and outputting into HDF5 format
def formatDataset(data_dir, series_names, res_out=None, outpath=None):
    # Loading the labels
    print("\nLoading the data")
    labels = []
    for sn in series_names:
        file_name = glob.glob(data_dir + sn + "_labels.csv")[0]
        lab = np.genfromtxt(file_name, dtype=float)
        print("Loaded {} labels for series {}".format(len(lab), sn))
        labels.append(lab)
    labels = np.concatenate(labels)

    # Loading the images
    file_list = [glob.glob(data_dir + ser_name + "_*.png") for ser_name in series_names]
    data = []
    for ind_ser in range(len(file_list)):
        count_img = 0
        for ind in range(len(file_list[ind_ser])):
            # Load an image in B&W
            fname = file_list[ind_ser][ind]
            img = cv2.imread(fname, 0)
            data.append(img)
            count_img += 1

        print("Loaded {} images for series {}".format(count_img, fname))

    # Loading the Haar cascade
    eye_cascade = cv2.CascadeClassifier('./config/haarcascade_eye.xml')
    if eye_cascade.empty():
        raise NameError("Couldn't find the Haar Cascade XML file.")

    # Loop information
    data_sz = data[0].shape
    faulty_data = 0

    dataLeftEye = []
    dataRightEye = []
    dataLeftCoord = []
    dataRightCoord = []
    dataLabels = []

    # Extracting the eye information
    print("Extracting the eye information")
    for i, img in tqdm(enumerate(data)):
        # Applying the Haar cascade
        eyes = eye_cascade.detectMultiScale(img)

        # Making sure we detect exactly 2 eyes
        try:
            assert (eyes.shape[0] == 2)
            # labels_valid.append(labels[i])
        except AssertionError:
            faulty_data += 1
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
        lec[0] /= w;
        lec[2] /= w;
        rec[0] /= w;
        rec[2] /= w
        lec[1] /= h;
        lec[3] /= h;
        rec[1] /= h;
        rec[3] /= h

        # Changing our coordinates to the center of the eye
        lec[0] += lec[2] / 2;
        lec[1] += lec[3] / 2
        rec[0] += rec[2] / 2;
        rec[1] += rec[3] / 2

        # Resizing to add the color channel for compatibility with tensorflow convolution layers
        le = le.reshape(res_out[0], res_out[1], 1)
        re = re.reshape(res_out[0], res_out[1], 1)

        # Appending to our data containers
        dataLeftEye.append(le)
        dataRightEye.append(re)
        dataLeftCoord.append(lec)
        dataRightCoord.append(rec)
        dataLabels.append(labels[i])

    # Printing the number of faulty images
    print("There were {} faulty images".format(faulty_data))

    # Converting to numpy arrays for HDF5 compatibility
    dataLeftEye = np.array(dataLeftEye)
    dataRightEye = np.array(dataRightEye)
    dataLeftCoord = np.array(dataLeftCoord)
    dataRightCoord = np.array(dataRightCoord)
    dataLabels = np.array(dataLabels)

    # Saving data to HDF5 file
    if outpath is not None:
        with h5py.File(outpath, 'w') as f:
            f.create_dataset("dataLeftEye", data=dataLeftEye)
            f.create_dataset("dataRightEye", data=dataRightEye)
            f.create_dataset("dataLeftCoord", data=dataLeftCoord)
            f.create_dataset("dataRightCoord", data=dataRightCoord)
            f.create_dataset("dataLabels", data=dataLabels)

    # Returning the data
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
    v = np.random.random((iter_num, 2))
    d1 = v[:, 0] - v[:, 1] # random guess
    d2 = v[:,0] - 0.5 # fixed guess of 0.5
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


if __name__ == "__main__":
    # To capture a dataset
    # create_data(num_expression=60, series_name="abc5", folder="./data/")

    # Formatting the dataset
    # data = formatDataset(data_dir="./data/",
    #                      series_names=['abc', 'abc1', 'abc2', 'abc3', 'abc4', 'abc5'],
    #                      res_out=(100, 100),
    #                      outpath="./data/compilation.h5")

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
    model = model_conv_eyes((100, 100, 1))
    save_complete_model(model=model,
                        inpath="./model/checkpoint.ckpt",
                        outpath="./model/savedmodels/1/")

    # img_gen = acquire_image()
    # eye_gen = extract_eye(img_gen, res_out=(100, 100))
    # webcam_gaze(modelpath="./model/savedmodels/1/", eye_gen=eye_gen)


    # Runnning the webcam application
    # webcam_smile(inpath="./model/model.h5", res_img=(64, 64))

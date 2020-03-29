import datetime
import argparse

import tensorflow as tf
import numpy as np
import cv2

import trainer
import data


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
    parser.add_argument("-l", "--load-dataset", action="store_true", help="Load dataset")
    parser.add_argument("-t", "--train-model", action="store_true", help="Train the model")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse inputs
    args = parse_inputs()

    # Check args for actions
    if args.collect_data:
        data.collect_data()

    if args.create_dataset:
        data.create_dataset()

    if args.load_dataset:
        dataset = data.load_dataset()
        left_eye, right_eye, left_eye_coordinates, right_eye_coordinates, labels = dataset

    if args.train_model:
        dataset = data.load_dataset()
        trainer.train_model(dataset)

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

import datetime
import glob
import os

from tqdm import tqdm
import numpy as np
import h5py
import cv2


def collect_data(num_expression=60, series_name=None, folder="./data/"):
    """
    Routine to collect new data. A

    :param num_expression:
    :param series_name:
    :param folder:
    :return:
    """

    # The resolution of the screen
    scr_width = 1600
    scr_height = 900

    # The half-size of the target
    target_size = 5

    # Set the series name
    if series_name is None:
        series_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

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
                      (x_pix - target_size, y_pix - target_size,),
                      (x_pix + target_size, y_pix + target_size),
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


def create_dataset(data_dir="./data",
                   series_names=None,
                   res_out=(100, 100),
                   outpath="./data/compilation.h5"):
    # Get the name of all the data series.
    if series_names is None:
        series_names = glob.glob(os.path.join(data_dir, "*.csv"))

    # Loading the labels.
    print("\nLoading the data")
    labels = []
    for name in series_names:
        label = np.genfromtxt(name, dtype=float)
        print("Loaded {} labels for series {}".format(len(label), name))
        labels.append(label)
    labels = np.concatenate(labels)

    # Loading the images
    file_list = [glob.glob(name[:-11] + "_*.png") for name in series_names]
    data = []
    for series in file_list:
        count_img = 0
        for name in series:
            # Load an image in B&W
            img = cv2.imread(name, 0)
            data.append(img)
            count_img += 1

        print("Loaded {} images for series {}".format(count_img, name))

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

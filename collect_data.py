import datetime
import glob

import numpy as np
import cv2


def create_data(num_expression=60, series_name=None, folder="./data/"):
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
    num_expression = 10
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

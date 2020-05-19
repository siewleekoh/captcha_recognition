import imutils
import cv2
import numpy as np

# helper function to resize image rray
def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image


# function to crop out white rows and resize
def img_process(np_im, size_x, size_y):
    non_white_rows = np.where(np.all(np_im == 255, axis=1) == False)
    first_element = non_white_rows[0][0] - 2  # the first row that is not white + 2 white rows
    last_element = non_white_rows[0][len(non_white_rows[0]) - 1] + 2  # the last row that is not white + 2 white rows
    non_white = np_im[first_element:last_element, :]
    resize_non_white = resize_to_fit(non_white, size_x, size_y)  # resize

    return resize_non_white
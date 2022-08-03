'''
Aux functions related to image processing
'''

# ---------------------------
# Imports
# ---------------------------

import cv2
import numpy as np
##from PIL import Image, ImageDraw, ImageFont  # Pillow!


# ---------------------------
# Functions
# ---------------------------

def view_image(image, wait=0):
    '''View image and optionally close it after 'wait' milliseconds'''

    # Create empty window
    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)

    # Resize window so it has the same aspect ratio as the image, but is the size of the window as it was last set.
    image_height, image_width = image.shape[:2]
    _, _, window_width, window_height = cv2.getWindowImageRect('Window')
    new_width = round(window_height * (image_width / image_height))
    cv2.resizeWindow('Window', new_width, window_height)

    # Show image
    cv2.imshow("Window", image)
    if wait:
        cv2.waitKey(wait)
        cv2.destroyAllWindows()
    else:
        cv2.waitKey()


def save_image(image, filename, verbose=False):
    if verbose:
        print_update(f'Saving file "{filename}"')
    # https://stackoverflow.com/a/44029918/3977107
    cv2.imwrite(str(filename), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def get_box(rectangle):
    x, y, w, h = rectangle
    return x, y, x+w, y+h


def get_area(rectangle):
    x0, y0, x1, y1 = rectangle
    w = x1 - x0
    h = y1 - y0
    return w * h


def convert_to_gray(image):
    '''Convert input to grayscale if it is not already so'''
    is_grayscale = len(image.shape) == 2
    return image.copy() if is_grayscale else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_gray_to_bgr(image):
    '''Convert input from grayscale to BGR color'''
    is_grayscale = len(image.shape) == 2
    assert is_grayscale
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def convert_bgr_to_rgb(image):
    '''Convert input from BGR to RGB'''
    is_grayscale = len(image.shape) == 2
    assert not is_grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def order_points(pts):
    # https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/#download-the-code

    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/#download-the-code

    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def grab_contours(cnts):
    # FROM: https://github.com/PyImageSearch/imutils

    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts

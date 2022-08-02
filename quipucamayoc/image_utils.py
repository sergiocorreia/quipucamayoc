'''
Aux functions related to image processing
'''

# ---------------------------
# Imports
# ---------------------------

import cv2
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

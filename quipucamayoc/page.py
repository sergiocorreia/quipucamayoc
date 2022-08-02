'''
Page class
'''

# ---------------------------
# Imports
# ---------------------------

import shutil
##import csv
##import copy
##import bisect
##import itertools
##from operator import attrgetter

import cv2
##from PIL import Image, ImageDraw, ImageFont  # Pillow!

##from .textbox import TextBox
##from .textbox_utils import is_garbage, fix_garbage, concatenate_words, concatenate_lines
##from .lines import load_lines_from_file
##from .table import Table

from .utils import *
from .image_utils import *

# ---------------------------
# Constants
# ---------------------------





# ---------------------------
# Ancillary functions
# ---------------------------

def debug_save(image, path, step):
    assert 0 <= step <= 100
    fn = path / f'{step}.jpg'
    save_image(image, fn)


# ---------------------------
# Main Class
# ---------------------------


class Page:

    def __init__(self, pagenum, filename, doc):
        assert 1 <= pagenum <= 9999
        self.pagenum = pagenum
        self.filename = filename
        self.doc = doc


    def load(self):
        self.image = cv2.imread(str(self.filename), cv2.IMREAD_UNCHANGED) #cv2.IMREAD_ANYDEPTH)


    def view(self, wait=0):
        '''View image and optionally close it after 'wait' milliseconds'''
        view_image(self.image, wait)


    def remove_black_background(self, threshold_parameter=25, verbose=False, debug=False):
        # TODO: either allow for custom min/max area/aspect ratios, or improve defaults
        
        if verbose:
            print_update(f' - Removing black background of page {self.pagenum}')

        if debug:
            debug_path = self.doc.cache_folder / 'tmp' / 'remove-black-background'
            shutil.rmtree(debug_path, ignore_errors = True) # Delete in case it already exists
            debug_path.mkdir()
            debug_save(self.image, debug_path, 0)

        # 1) Ensure the input is in grayscale
        is_grayscale = len(self.image.shape) == 2
        if verbose:
            print_update(f'   - Color mode: {"grayscale" if is_grayscale else "color"}')
        im = self.image.copy() if is_grayscale else cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if debug:
            debug_save(im, debug_path, 1)

        height, width = im.shape[:2]
        page_area = height * width

        # 2) Apply simple threshold (performed better than adaptive ones)
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
        #_, im = cv2.threshold(im, threshold_parameter, 255, cv2.THRESH_BINARY)
        _, im = cv2.threshold(im, threshold_parameter, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if debug:
            debug_save(im, debug_path, 2)

        # 3) Find contours
        contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if verbose:
            if contours:
                print_update(f'   - {len(contours)} contours found')
            else:
                print_update( f'   - [red]Failed to find contours')


        rectangles = [cv2.boundingRect(contour) for contour in contours]
        rectangles = [(x[2]*x[3], get_box(x)) for x in rectangles]  # Add area
        rectangles = sorted(rectangles)  # Sort by area
        largest_rectangle = rectangles[-1][1]
        
        area = rectangles[-1][0]
        area_ratio = area / page_area

        x0, y0, x1, y1 = largest_rectangle
        aspect_ratio = (y1-y0) / (x1-x0)

        ok = (1.2 <= aspect_ratio <= 1.6) and (0.7 <= area_ratio <= 0.95)
        if verbose:
            print_update(f'   - Method worked? {ok}')
            print_update(f'     aspect ratio (height/width): {aspect_ratio*100:4.1f}%')
            print_update(f'     area ratio (rectangle area / page area):  {area_ratio*100:4.1f}%')
        
        if debug:
            line_width = max(5, int(height / 200))  # Max between 5 pixels and 0.5% of the image height
            im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
            im = cv2.rectangle(im, (x0, y0), (x1, y1), (0,255,0), line_width)
            debug_save(im, debug_path, 3)
        
        # Update answer
        if ok:
            self.image = self.image[y0:y1, x0:x1]
            if debug:
                debug_save(self.image, debug_path, 4)

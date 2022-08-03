'''
Page class

TODO:
    - remove_fore_edges() could be improved with automatic threshold selection,
      by looking at the inflection point of the histogram (already computed)
    - remove_fore_edges() might not work if the book edge is too light; as
      an alternative we might want to try to use the line detection algo. (canny)
    - remove_fore_edges() might want to exploit the fact that page sizes often have
      very specific aspect ratios.
'''

# ---------------------------
# Imports
# ---------------------------

##import csv
##import copy
##import bisect
##import itertools
##from operator import attrgetter

import cv2
import numpy as np
##from PIL import Image, ImageDraw, ImageFont  # Pillow!

# Only installed with pip install quipucamayoc[dev]
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

##from .textbox import TextBox
##from .textbox_utils import is_garbage, fix_garbage, concatenate_words, concatenate_lines
##from .lines import load_lines_from_file
##from .table import Table

from .utils import *
from .image_utils import *


from .dewarpnet import dewarpnet

# ---------------------------
# Constants
# ---------------------------





# ---------------------------
# Ancillary functions
# ---------------------------

def debug_save(image, path, step):
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


    def unload(self):
        self.image = None


    def save(self, verbose=False, debug=False):
        if verbose:
            print_update(f' -  Saving image to disk ({self.filename})')
        save_image(self.image, self.filename, verbose=False)


    def view(self, wait=0):
        '''View image and optionally close it after 'wait' milliseconds'''
        view_image(self.image, wait)


    def rotate(self, angle=-90, verbose=False, debug=False):
        if verbose:
            print_update(f' - Rotating page {angle} degrees')
        self.image = self.image.rotate(angle, expand=True)


    def remove_black_background(self, threshold_parameter=25, verbose=False, debug=False):
        # TODO: either allow for custom min/max area/aspect ratios, or improve defaults
        
        if verbose:
            print_update(f' - Removing black background of page {self.pagenum}')

        if debug:
            debug_path = create_folder(self.doc.cache_folder / 'tmp' / 'remove-black-background', delete_before=True, try_again=True)
            debug_save(self.image, debug_path, 0)

        # 1) Ensure the input is in grayscale
        im = convert_to_gray(self.image)
        height, width = im.shape[:2]
        page_area = height * width
        if debug:
            debug_save(im, debug_path, 1)


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


    def remove_fore_edges(self, threshold_parameter=160, verbose=False, debug=False):

        if verbose:
            print_update(f' - Removing fore edges of page {self.pagenum}')

        if debug:
            debug_path = create_folder(self.doc.cache_folder / 'tmp' / 'remove-fore-edges', delete_before=True, try_again=True)
            debug_save(self.image, debug_path, '0-original')

        # 1) Gray version
        im = convert_to_gray(self.image)
        height, width = im.shape[:2]
        page_area = height * width
        #im = self.image[:,:,0]  # Extract a single channel from BGR # OFTEN USING A SINGLE CHANNEL (BLUE) IS BETTER WITH YELLOWISH DOCUMENTS
        if debug:
            debug_save(im, debug_path, '1-grayscale')

        # Calculate histogram (so we can tweak the threshold parameter)
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html
        # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
        if debug:
            #hist = cv2.calcHist([im], [0], None, [256], [0,256])
            hist = np.bincount(im.ravel(), minlength=256)
            hist_fn = debug_path / 'histogram.txt'
            np.savetxt(str(hist_fn), hist, newline='\n', fmt='%i') # fmt='%10d',

            # https://stackoverflow.com/a/27084005/3977107
            #fig, ax = plt.subplots()
            #ax.bar(range(256), hist, width=1, align='center')
            #ax.set(xticks=np.arange(0, 257, 32), xlim=[0, 256])
            #plt.show()

            # Slower than reusing bincount but we only do this once (if at all)
            plt.hist(im.ravel(), bins=255, range=(0,255), density=True, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='none', cumulative=True)
            plt.ylabel('Density')
            plt.xlabel('Grayscale (0=black 255=white)')
            plt.xticks(np.arange(0, 257, 32))
            plt.axvline(x=threshold_parameter, ymin=0, color='red', linestyle='dotted', linewidth=3, alpha=0.5)
            hist_fn = debug_path / 'histogram.png'
            plt.savefig(hist_fn)


        im = cv2.GaussianBlur(im,(5,5),0)
        _, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


        # 3) Apply simple threshold (performed better than adaptive ones)
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
        #threshold_parameter = 159 # 127
        _, thresh = cv2.threshold(im, threshold_parameter, 255, cv2.THRESH_BINARY)
        #_, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Doesn't work that well (e.g. 1916 p1000)
        if debug:
            debug_save(thresh, debug_path, '2-threshold')

        # 4) Remove noise (opening = erosion followed by dilation)
        kernel_size = 21
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel, iterations = 2)
        if debug:
            debug_save(opening, debug_path, '3-denoise')

        # 5) Expand
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        sure_bg = cv2.dilate(opening,kernel,iterations=2)
        if debug:
            debug_save(sure_bg, debug_path, '4-expand')

        # 6) Find contours
        # Tutorial on hierarchy: https://vovkos.github.io/doxyrest-showcase/opencv/sphinxdoc/page_tutorial_py_contours_hierarchy.html
        contours, hierarchy = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if debug:
            contours2, hierarchy2 = cv2.findContours(sure_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            im2 = cv2.drawContours(self.image, contours2, -1, (120,120,0), 6)
            im2 = cv2.drawContours(im2, contours, -1, (0,255,0), 16)
            # Arguments:
            # = image:      Destination image.
            # = contours:   All the input contours. Each contour is stored as a point vector.
            # = contourIdx: Parameter indicating a contour to draw. If it is negative, all the contours are drawn.
            # = color:      Color of the contours.
            # = thickness:  Thickness of lines the contours are drawn with. If it is negative, the interiors of the contour are filled with the colour specified.
            debug_save(im2, debug_path, '5-contours')
            print(len(contours))

        # 7) Find largest rectangle
        # - If there are no rectangles, abort
        # - If two rectangles are larger than 10%, merge them
        # - If final rectangle is less than 50% of the page's area, abort
        ok = False
        rectangles = []

        if not contours:
            print_update('   - No contours detected; [red]stopping')
            return

        rectangles = [get_box(cv2.boundingRect(contour)) for contour in contours]
        
        if verbose:
            print_update(f'   - {len(rectangles)} rectangles found!')
        
        if debug:
            im2 = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
            #largest_rectangle = sorted(rectangles, key=get_area)[-1]
            line_width = max(5, int(height / 200))  # Max between 5 pixels and 0.5% of the image height
            for (x0, y0, x1, y1) in rectangles:
                im2 = cv2.rectangle(im2, (x0, y0), (x1, y1), (255,0,0), line_width)  # Recall opencv uses BGR instead of RGB
                #x0, y0, x1, y1 = largest_rectangle
            #cv2.rectangle(im2, (x0, y0), (x1, y1), (0,255,0), 40)
            debug_save(im2, debug_path, '6-rectangles')

        # Get large rectangles
        rectangles = [x for x in rectangles if get_area(x) > 0.1 * page_area]
        if not rectangles:
            print_update('   - No large-enough rectangles detected; [red]stopping')
            return

        if verbose:
            print_update(f'   - {len(rectangles)} rectangles are large enough')

        # Combine all large rectangles
        x0s, y0s, x1s, y1s = zip(*rectangles)
        x0, y0 = min(x0s), min(y0s)
        x1, y1 = max(x1s), max(y1s)
        rectangle = (x0, y0, x1, y1)

        if get_area(rectangle) < 0.5 * page_area:
            print_update(f'   - Largest rectangle smaller than 50% of original page; [red]stopping')
            return

        if get_area(rectangle) == page_area:
            print_update(f'   - Largest rectangle is same size as page; [red]stopping')
            return

        if verbose:
            print_update(f'   - Method worked? {ok}')

        if debug:
            line_width = max(20, int(height / 100))  # Max between 20 pixels and 1% of the image height
            cv2.rectangle(im2, (x0, y0), (x1, y1), (0,255,0), line_width)
            debug_save(im2, debug_path, '7-selection')

        trim = 10
        y0, x0 = y0+trim, x0+trim
        y1, x1 = y1-trim, x1-trim
        self.image = self.image[y0:y1, x0:x1]
        
        if verbose:
            perc = 100 * get_area(rectangle) / page_area
            aspect_ratio = (y1-y0) / (x1-x0)
            print_update(f' - Image fixed ({perc:5.2f}% of page area)')
            print_update(f'   - New aspect ratio is {aspect_ratio:4.2F} vs {height/width:4.2f} of page')
        
        if debug:
            debug_save(self.image, debug_path, '8-results')


    def dewarp(self, method='simple',
            wc_model_path=None, bm_model_path=None,
            verbose=False, debug=False):

        assert method in ('simple', 'dewarpnet')

        if verbose:
            print_update(f' - Dewarping page {self.pagenum} (method={method})')

        if debug:
            debug_path = create_folder(self.doc.cache_folder / 'tmp' / 'dewarp', delete_before=True, try_again=True)
            debug_save(self.image, debug_path, '0-original')

        if method=='simple':
           dewarped, ok = self._dewarp_simple(verbose, debug)
        elif method=='dewarpnet':
            if wc_model_path is None:
                error_and_exit('wc_model_path is None')
            dewarped, ok = dewarpnet(self.image,
                wc_model_path=wc_model_path, bm_model_path=bm_model_path,
                verbose=verbose, debug=debug)

        if verbose:
            print_update(f'   - Dewarp worked? {ok}')

        if ok:
            if verbose:
                print_update(f'   - Updating image with dewarped version')
            self.image = dewarped

            if debug:
                debug_save(self.image, debug_path, '9-results')
    

    def _dewarp_simple(self, verbose=False, debug=False):
        # https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
        # TODO: Avoid false positives?
        # TODO: allow resizing as in blog?

        if debug:
            debug_path = self.doc.cache_folder / 'tmp' / 'dewarp'  # Folder already created by dewarp()

        im = convert_to_gray(self.image)
        height, width = im.shape[:2]
        page_area = height * width

        if verbose:
            print_update(f'   - Applying gaussian blur')
        im = cv2.GaussianBlur(im, (5, 5), 0)

        if verbose:
            print_update(f'   - Applying canny edge detector')
        edged = cv2.Canny(im, 75, 200)
        if debug:
            debug_save(im, debug_path, '1-edges')

        # BUGBUG: This doesn't seem to be that robust to just selecting tables or other bugs...
        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        if verbose:
            print_update(f'   - Detecting paper contour')
        contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = grab_contours(contours)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
        # loop over the contours
        for c in contours:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                break
        # show the contour (outline) of the piece of paper
        if debug:
            im2 = cv2.drawContours(self.image.copy(), [screenCnt], -1, (0, 255, 0), 2)
            debug_save(im2, debug_path, '2-contour')

        points = screenCnt.reshape(4, 2)
        warped = four_point_transform(self.image, points)
        new_height, new_width = warped.shape[:2]
        new_page_area = new_height * new_width

        if new_page_area < 0.7 * page_area:
            print_update(f'   - New page area is too small ({100*new_page_area/page_area:4.1}% of original); [red]stopping')
            return None, False

        #if new_page_area == page_area:
        #    print_update(f'   - New page area is the same as old; [red]stopping')
        #    return


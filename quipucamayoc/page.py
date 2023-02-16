'''
Page class

TODO:
    - remove_fore_edges() could be improved with automatic threshold selection,
      by looking at the inflection point of the histogram (already computed)
    - remove_fore_edges() might not work if the book edge is too light; as
      an alternative we might want to try to use the line detection algo. (canny)
    - remove_fore_edges() might want to exploit the fact that page sizes often have
      very specific aspect ratios.
    - Dewarp ML functions could be implemented as a Class to save state;
      else they will be very slow with multiple pages as they have to load the model
      each time.
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
import cv2.ximgproc
import numpy as np
from rich.console import Console
from rich.table import Table
##from PIL import Image, ImageDraw, ImageFont  # Pillow!

# Only installed with pip install quipucamayoc[dev]
#try:
#    import matplotlib.pyplot as plt # Very slow so imported directly in function
#except ImportError:
#    pass

##from .textbox import TextBox
##from .textbox_utils import is_garbage, fix_garbage, concatenate_words, concatenate_lines
##from .lines import load_lines_from_file
##from .table import Table

from .utils import *
from .image_utils import *
# from .DewarpNet import DewarpNet # Very slow so imported directly in function
# from .DocTr import DocTr # Very slow so imported directly in function

from .ocr_aws import run as run_ocr_aws


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


    def describe(self):
        if self.image is None: error_and_exit('image not loaded')
        
        shape = self.image.shape
        assert len(shape) in (2, 3)
        height = shape[0]
        width = shape[1]
        is_color = image_is_color(self.image)
        size = f'{self.image.size//1024}KB'
        
        if (height * 4 % 300 == 0) and (width * 4 % 300 == 0):
            dpi = '300'
        elif (height * 4 % 72 == 0) and (width * 4 % 72 == 0):
            dpi = '72'
        else:
            dpi = 'unknown'

        ratio = height / width
        page_ratios = {'Arch A': 1.3333, 'Letter': 1.2941, 'Letter Wide': 0.7727, 'Legal': 1.6471, 'Legal Wide': 0.6071, 'A3': 1.4141, 'A4': 1.4143, 'A4 Wide': 0.7071, 'A5': 1.4189, 'B4': 1.4163, 'B4 Wide': 0.7060, 'B5': 1.4121, 'Folio': 1.5294, 'Ledger': 0.6471, 'Tabloid': 1.5455, 'Quarto': 1.2791, 'Short': 1.2353, 'Statement': 1.5455, 'Stationery': 1.2500}
        candidate = sorted([(abs(v-ratio), k) for k, v in page_ratios.items()])[0]
        if candidate[0] > 0.05:
            page_size = 'unknown'
        else:
            page_size = f'{candidate[1]} (err={candidate[0]:3.1f})'

        console = Console()
        table = Table(show_header=True, header_style="bold cyan dim", title='Page Metadata', style='cyan dim')
        table.add_column("Key", min_width=14, style='cyan')
        table.add_column("Value", justify="left", style='cyan')
        table.add_row('Page number', str(self.pagenum))
        table.add_row('Dimension', f'{height}x{width} ({ratio:5.4})')
        table.add_row('Color', str(is_color))
        table.add_row('Size', size)
        table.add_row('Dtype', str(self.image.dtype))
        table.add_row('DPI guess', dpi)
        table.add_row('Page size guess', page_size)
        #if self.title:
        #    table.add_row('Title', self.title)
        #if self.author:
        #    table.add_row('Author', self.author)
        #if self.creator:
        #    table.add_row('Creator', self.creator)
        #if self.producer:
        #    table.add_row('Producer', self.producer)
        #if self.creation_date:
        #    table.add_row('Creation date', self.creation_date)
        #table.add_row('Size', f'{self.size / 2 ** 20:<6.2f}MiB')
        #table.add_row('Num pages', f'{self.num_pages}')
        #table.add_row('Active pages', f'{self.first_page}-{self.last_page} ({self.last_page-self.first_page+1} pages)')
        #table.add_row('Cache folder', str(self.cache_folder))
        print()
        console.print(table)
        print()


    def is_grayscale(self):
        if self.image is None: error_and_exit('image not loaded')
        return image_is_grayscale(self.image)


    def convert_to_grayscale(self):
        if self.image is None: error_and_exit('image not loaded')
        self.image = convert_image_to_gray(self.image)


    def load(self):
        self.image = cv2.imread(str(self.filename), cv2.IMREAD_UNCHANGED) #cv2.IMREAD_ANYDEPTH)


    def unload(self):
        self.image = None


    def save(self, debug_name=None, verbose=False, debug=False):
        if self.image is None: error_and_exit('image not loaded')
        # debug_name allows us to save it to a different location for comparison/debugging purposes
        filename = (self.doc.cache_folder / debug_name) if debug_name is not None else self.filename
        if verbose:
            print_update(f' -  Saving image to disk ({filename})')
        save_image(self.image, filename, verbose=False)


    def view(self, wait=0):
        '''View image and optionally close it after 'wait' milliseconds'''
        if self.image is None: error_and_exit('image not loaded')
        if is_notebook():
            from IPython.display import display
            from PIL import Image
            display(Image.fromarray(self.image))
        else:
            view_image(self.image, wait)


    def rotate(self, angle=-90, verbose=False, debug=False):
        if self.image is None: error_and_exit('image not loaded')
        if verbose:
            print_update(f' - Rotating page {angle} degrees')
        self.image = self.image.rotate(angle, expand=True)


    def remove_black_background(self, threshold_parameter=25, verbose=False, debug=False):
        if self.image is None: error_and_exit('image not loaded')

        # TODO: either allow for custom min/max area/aspect ratios, or improve defaults
        
        if verbose:
            print_update(f' - Removing black background of page {self.pagenum}')

        if debug:
            debug_path = create_folder(self.doc.cache_folder / 'tmp' / 'remove-black-background', delete_before=True, try_again=True)
            debug_save(self.image, debug_path, 0)

        # 1) Ensure the input is in grayscale
        im = convert_image_to_gray(self.image)
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
        if self.image is None: error_and_exit('image not loaded')

        if verbose:
            print_update(f' - Removing fore edges of page {self.pagenum}')

        if debug:
            debug_path = create_folder(self.doc.cache_folder / 'tmp' / 'remove-fore-edges', delete_before=True, try_again=True)
            debug_save(self.image, debug_path, '0-original')

        # 1) Gray version
        im = convert_image_to_gray(self.image)
        height, width = im.shape[:2]
        page_area = height * width
        #im = self.image[:,:,0]  # Extract a single channel from BGR # OFTEN USING A SINGLE CHANNEL (BLUE) IS BETTER WITH YELLOWISH DOCUMENTS
        if debug:
            debug_save(im, debug_path, '1-grayscale')

        # Calculate histogram (so we can tweak the threshold parameter)
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html
        # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
        if debug:
            import matplotlib.pyplot as plt # Only installed with pip install quipucamayoc[dev]
            
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
            #print(len(contours))

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


    def dewarp(self, method='simple', model_path=None,
            rectify_illumination=False,
            verbose=False, debug=False):

        '''Dewarp function; calling alternative dewarp methods.

        For a more comprehensive list of dewarp methods, see:
        https://github.com/fh2019ustc/Awesome-Document-Image-Rectification

        Within that list, these packages have github links:
        - DocProj               : [✗] uses an .exe file for stitching
        - DewarpNet             : [✓] implemented
        - ..displacement flow.. : [✗] couldn't get inference to run
        - DocTr                 : [✓] implemented
        - PiecewiseUnwarp       : [✗] no code yet
        - ..control points..    : [✗] couldn't get inference to run
        - Marior                : [✗] no code yet
        - PaperEdge             : [✗] new; code and pre-trained data partly there

        Also:
        - PageDewarp https://github.com/lmmx/page-dewarp (works quite well for some cases)
        - https://safjan.com/tools-for-doc-deskewing-and-dewarping/
        '''

        method = method.lower()  # Allows for DewarpNet, DocTr, etc.
        assert method in ('simple', 'dewarpnet', 'doctr')
        if self.image is None: error_and_exit('image not loaded')

        if verbose:
            print_update(f' - Dewarping page {self.pagenum} (method={method})')

        if debug:
            debug_path = create_folder(self.doc.cache_folder / 'tmp' / 'dewarp', delete_before=True, try_again=True)
            debug_save(self.image, debug_path, '0-original')

        if method == 'simple':
           dewarped, ok = self._dewarp_simple(verbose, debug)
        elif method == 'dewarpnet':
            if model_path is None:
                error_and_exit('model_path is None')
            from .DewarpNet import DewarpNet # Import here to avoid slowing down the main code
            dewarped, ok = DewarpNet(self.image, model_path, self.doc.models, verbose=verbose, debug=debug)
        elif method == 'doctr':
            if model_path is None:
                error_and_exit('model_path is None')
            from .DocTr import DocTr
            dewarped, ok = DocTr(self.image, model_path, self.doc.models, rectify_illumination=rectify_illumination, verbose=verbose, debug=debug)
        else:
            raise Exception

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

        im = convert_image_to_gray(self.image)
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

        if not contours:
            print_update(f'   - No contours detected; [red]stopping')
            return None, False
        
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
        else:
            print_update(f'   - No approximated contour detected with four points; [red]stopping')
            return None, False

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


    def equalize(self, method='clahe',
            clip_limit=2, tile_grid_size=8,
            verbose=False, debug=False):
        '''Equalize grayscale image to improve its contrast

        Methods: 
        - Simple
        - CLAHE
        '''

        method = method.lower()
        assert method in ('simple', 'clahe')
        if self.image is None: error_and_exit('image not loaded')
        if not self.is_grayscale(): error_and_exit('image must be in grayscale before equalizing; use .convert_to_grayscale()')

        if verbose:
            print_update(f' - Equalizing page {self.pagenum} (method={method})')

        if debug:
            debug_path = create_folder(self.doc.cache_folder / 'tmp' / 'equalize', delete_before=True, try_again=True)
            plt.hist(self.image.ravel(), 256, [0,256])
            plt.savefig(debug_path / 'histogram-original.pdf')
            plt.clf()
            debug_save(self.image, debug_path, '0-original')

        if method == 'simple':
            self.image = cv2.equalizeHist(self.image)
        elif method == 'clahe':
            # https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
            self.image = clahe.apply(self.image)
        else:
            raise Exception

        if debug:
            plt.hist(self.image.ravel(), 256, [0,256])
            plt.savefig(debug_path / f'histogram-{method}.pdf')
            plt.clf()
            debug_save(self.image, debug_path, f'1-{method}')


    def binarize(self, method='wolf',
            threshold=128,
            block_size=11, k=0.1,
            verbose=False, debug=False):
        '''Binarize grayscale image to improve its contrast

        Methods: 
        - Simple or Threshold
        - Otsu
        - Adaptive mean (adaptive)
        - Sauvola
        - Wolf
        '''

        method = method.lower()
        if method == 'simple': method == 'threshold'
        assert method in ('threshold', 'otsu', 'adaptive', 'sauvola', 'wolf')

        if self.image is None: error_and_exit('image not loaded')
        if not self.is_grayscale(): error_and_exit('image must be in grayscale before equalizing; use .convert_to_grayscale()')

        if verbose:
            print_update(f' - Binarizing page {self.pagenum} (method={method})')

        if debug:
            debug_path = create_folder(self.doc.cache_folder / 'tmp' / 'binarize', delete_before=True, try_again=True)
            debug_save(self.image, debug_path, '0-original')

        if method == 'threshold':
            _, self.image = cv2.threshold(self.image, threshold_parameter, 255, cv2.THRESH_BINARY)
        elif method == 'otsu':
            _, self.image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Doesn't work that well
        elif method == 'adaptive':
            # https://stackoverflow.com/questions/28763419/adaptive-threshold-parameters-confusion/28764902
            # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
            # blockSize – Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
            # C – Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.
            
            # Two alternatives: adaptive GAUSSIAN and adaptive MEAN
            #self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,5)
            self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,5)
        elif method == 'sauvola':
            # https://shimat.github.io/opencvsharp_docs/html/9b2f295b-eb64-b5e8-ba39-33cbe88d5b4e.htm
            # blockSize: Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
            # k: The user-adjustable parameter used by Niblack and inspired techniques.For Niblack, this is normally a value between 0 and 1 that is multiplied with the standard deviation and subtracted from the mean.
            self.image = cv2.ximgproc.niBlackThreshold(self.image,
                    maxValue=255,
                    type=cv2.THRESH_BINARY_INV,
                    blockSize=block_size,
                    k=k,
                    binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)
            self.image = (255 - self.image)
        elif method == 'wolf':
            self.image = cv2.ximgproc.niBlackThreshold(self.image,
                    maxValue=255,
                    type=cv2.THRESH_BINARY_INV,
                    blockSize=block_size,
                    k=k,
                    binarizationMethod=cv2.ximgproc.BINARIZATION_WOLF)
            self.image = (255 - self.image)
        else:
            raise Exception

        if debug:
            debug_save(self.image, debug_path, f'1-{method}')


    def run_ocr(self, engine=None, extract_tables=False, verbose=False, debug=False):

        if engine is None: error_and_exit('you must specify an OCR engine (aws, gcv, etc.)')
        engine = engine.lower()
        if engine in  ('amazon', 'textract'): engine = 'aws'
        if engine in  ('google', 'visionai', 'cloudvision'): engine = 'gcv'
        if not engine in ('aws', 'gcv'): error_and_exit(f'engine {engine} is unsupported; supported engines are: aws, gcv')

        if engine == 'aws':
            run_ocr_aws(self, extract_tables=extract_tables, verbose=verbose, debug=debug)
        else:
            raise Exception

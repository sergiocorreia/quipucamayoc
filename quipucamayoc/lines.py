"""
Based on:
- https://github.com/WZBSocialScienceCenter/pdftabextract/blob/master/pdftabextract/imgproc.py
"""

# ---------------------------
# Imports
# ---------------------------

import os
import sys
import math
from pathlib import Path

import numpy as np
import cv2 # pip install opencv-python
import pdftabextract.imgproc # pip install pdftabextract
from pdftabextract.geom import project_polarcoord_lines

#from utils import *


# ---------------------------
# Constants
# ---------------------------

PIHLF = np.pi / 2
PI4TH = np.pi / 4
CANNY_LOW_THRESH = 100
CANNY_HIGH_THRESH = 200
APERTURE_SIZE = 5 # canny_kernel_size.. 3?
HOUGH_RHO_RES = 1.0 # 1.0 # LOWER??
HOUGH_THETA_RES = np.pi / 180 # np.pi / 600 // 180 


MAX_LINE_DIST = 40  # Used when grouping lines


# ---------------------------
# Main Function
# ---------------------------

def detect_lines_in_page(page, image_fn, lines_fn, out_image_fn, num_hlines=None, num_vlines=None):
    
    print(f"   - Detecting lines in page {page}: '{image_fn}'")

    # 1) Read image
    img = cv2.imread(str(image_fn))
    height, width = img.shape[:2]

    # 2) Denoise
    img = cv2.medianBlur(img, 3) # aperture size (must be odd and greater than 1)
    #kernel = np.ones((7, 7), np.uint8)
    #img = cv2.dilate(img, kernel, iterations = 1)
    #img = cv2.erode(img, kernel, iterations = 1)

    # 3) Convert to grayscale and then apply threshold to convert to B&W
    # https://docs.opencv.org/3.3.1/d7/d4d/tutorial_py_thresholding.html
    tmp_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Do we need this?? check...

    # 4) Compute optimal threshold (OTSU)
    high_threshold, _ = cv2.threshold(tmp_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_threshold = 0.5 * high_threshold
    #high_threshold, low_threshold = CANNY_HIGH_THRESH * 2, CANNY_LOW_THRESH / 2

    # 5) Detect edges
    edges = cv2.Canny(tmp_img, low_threshold, high_threshold, apertureSize=APERTURE_SIZE, L2gradient=True)
    #edges = cv2.Canny(image, CANNY_LOW_THRESH, CANNY_HIGH_THRESH, apertureSize=APERTURE_SIZE, L2gradient=True)

    # 6) Get horizontal lines
    if num_hlines is not None: print(f"   - Expecting {num_hlines} horizontal lines")
    min_hlines = None if num_hlines is None else num_hlines
    max_hlines = None if num_hlines is None else num_hlines
    hlines, hcandidates = optimizer(edges_image=edges, height=height, width=width,
                                    is_horizontal=True, min_lines=min_hlines, max_lines=max_hlines)

    # 7) Get vertical lines
    if num_vlines is not None: print(f"   - Expecting {num_vlines} vertical lines")
    min_vlines = None if num_vlines is None else num_vlines
    max_vlines = None if num_vlines is None else num_vlines
    vlines, vcandidates = optimizer(edges_image=edges, height=height, width=width,
                                    is_horizontal=False, min_lines=min_vlines, max_lines=max_vlines)

    # 8) Save debugging image
    if out_image_fn is not None:
        debug_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        hcandidates = set(tuple(_.tolist()) for _ in hcandidates)
        vcandidates = set(tuple(_.tolist()) for _ in vcandidates)
        candidates = list(hcandidates | vcandidates) # convert to set and back to remove dupes

        for x0, y0, x1, y1 in candidates:
            xy0 = (x0, y0)
            xy1 = (x1, y1)
            cv2.line(debug_img, xy0, xy1, (100, 100, 200), 2, cv2.LINE_AA) # 8

        for y in hlines:
            xy0 = (0, y)
            xy1 = (width, y)
            cv2.line(debug_img, xy0, xy1, (0, 160, 0), 12, 8) # 8

        for x in vlines:
            xy0 = (x, 0)
            xy1 = (x, height)
            cv2.line(debug_img, xy0, xy1, (160, 0, 0), 12, 8) # 8

        cv2.imwrite(str(out_image_fn), debug_img)

    # 9) Save lines as text file
    hlines = ' '.join(str(_) for _ in hlines)
    vlines = ' '.join(str(_) for _ in vlines)
    out = '\n'.join((hlines, vlines))

    with open(lines_fn, 'w') as f:
            f.write(out)


def optimizer(min_lines, max_lines, **kwargs):
    adjustment = 1
    lines, candidates = inner(**kwargs, adjustment=adjustment)
    n = len(lines)
    used_optimizer = False

    i = 0
    while (min_lines is not None) and (n < min_lines):
        used_optimizer = True
        adjustment = adjustment * 1.1
        #print(n, min_lines, adjustment)
        i += 1
        if i > 5:
            print(f'     [optimizer] too many tries')
            break
        lines, candidates = inner(**kwargs, adjustment=adjustment)
        n = len(lines)

    if used_optimizer:
        print(f'     [optimizer] {len(lines)} lines detected in {i+1} attempts')
    return lines, candidates



def inner(edges_image, height, width, is_horizontal, adjustment=1):
    msg = 'horizontal' if is_horizontal else 'vertical'
    
    # http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny
    # http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny#houghlines

    hough_votes_thresh = round(0.05 * min(width, height))
    #hough_votes_thresh = 80
    #hough_votes_thresh = round(0.28 * width)

    print('   - Hough voting threshold:', hough_votes_thresh)
    candidates = cv2.HoughLinesP(edges_image,
                                 rho=HOUGH_RHO_RES,
                                 theta=HOUGH_THETA_RES,
                                 threshold=hough_votes_thresh,
                                 minLineLength=150/adjustment, # minimum number of points that can form a line
                                 maxLineGap=5)  # max gap between two points to be considered in the same line.

    if candidates is None:
        print(f'No {msg} lines detected!!!')
        return [], []
    else:
        print(f'     {len(candidates)} {msg} lines detected')

    for c in candidates:
        assert len(c) == 1
    
    candidates = [c[0] for c in candidates]
    lines = []

    for x0, y0, x1, y1 in candidates:
        # I can use this to tweak the rotation
        angle = abs(math.atan2(y1 - y0, x1 - x0) * 180 / np.pi) # * 180 / CV_PI

        xy0 = (x0, y0)
        xy1 = (x1, y1)

        if abs(angle - 90) < (2 * adjustment):
            if not is_horizontal:
                x = int(round((x0 + x1) / 2))
                lines.append(x)
        elif abs(angle - 0) < (2 * adjustment):
            if is_horizontal:
                y = int(round((y0 + y1) / 2))
                lines.append(y)
        else:
            print(f'{msg} line ignored:', xy0, xy1, angle)

    lines = cluster_lines(lines)
    print(f'     {len(lines)} {msg} lines confirmed')
    return lines, candidates


def cluster_lines(lines):
    lines = sorted(lines)
    newlines = []
    last = None # First and most recent lines in the cluster
    k = None

    for x in lines:
        if last is None:
            last = x
            k = 1
        else:
            avg = (k * last + x) / (k + 1)  # weighted avg
            if (x - avg) < MAX_LINE_DIST:
                last = avg
                k += 1
            else:
                newlines.append(last)
                last = x
                k = 1

    if last is not None:
        newlines.append(last)

    newlines = [int(round(x)) for x in newlines]
    return newlines


def load_lines_from_file(fn):
    with open(fn, 'r') as fh:
        hlines, vlines = [[int(x) for x in row.strip().split()] for row in fh.readlines()]
    return hlines, vlines



# UNUSED....









        #fh.write("\n<EOF>\n")

    #img = pdftabextract.imgproc.ImageProc(tif_fn)
    
    # calculate the scaling of the image file in relation to the text boxes coordinate system dimensions
    #print(img.width, img.height)
    ##page_scaling_x = img.width / p['width']   # scaling in X-direction
    ##page_scaling_y = img.height / p['height']  # scaling in Y-direction

    # detect the lines
    #lines_hough = img.detect_lines(canny_low_thresh=100, # 50
    #                               CANNY_HIGH_THRESH=200, # 150
    #                               canny_kernel_size=3,
    #                               HOUGH_RHO_RES=1,
    #                               HOUGH_THETA_RES=np.pi/150, # /300
    #                               hough_votes_thresh=round(0.28 * img.width))










def ab_lines_from_hough_lines(lines_hough, height, width):
    """
    From a list of lines <lines_hough> in polar coordinate space, generate lines in cartesian coordinate space
    from points A to B in image dimension space. A and B are at the respective opposite borders
    of the line projected into the image.
    Will return a list with tuples (A, B, DIRECTION_HORIZONTAL or DIRECTION_VERTICAL).
    """
    
    projected = project_polarcoord_lines([l[:2] for l in lines_hough], width, height)
    return [(p1, p2, line_dir) for (p1, p2), (_, _, _, line_dir) in zip(projected, lines_hough)]

def avg_dim(line, dim):
    return int((line[0][dim]+line[1][dim])/2)

def fix_lines(lines, page):
    hlines = sorted(avg_dim(l, 1) for l in lines if l[-1]=='h')
    vlines = sorted(avg_dim(l, 0) for l in lines if l[-1]=='v')

    hlines = cluster_lines(hlines)
    vlines = cluster_lines(vlines)
    
    #print(len(hlines), len(vlines))
    #hlines = probable_hlines(hlines)
    #vlines = probable_vlines(vlines)
    #print(len(hlines), len(vlines))
    #ok = len(hlines)==7 and len(vlines)==3

    #if not ok:
    #    print('Probable error on page {}'.format(page))

    return hlines, vlines #, ok


def probable_hlines(lines):
    newlines = []
    for i, line in enumerate(lines, 1):
        if (len(newlines)==0) and (line < 130) and len(lines)>=i+7: # 368 975
            if lines[i+1]<=400:
                continue
        elif (len(newlines) >= 2) and (380 < line < 960): # 368 975
            continue
        elif (len(newlines) >= 4) and (1105 < line < 1680): # 1095 1694
            continue
        elif (len(newlines) >= 6) and (1840 < line < 2400): # 1820 2425
            continue
        newlines.append(line)
    return newlines


def probable_vlines(lines):
    newlines = []
    for i, line in enumerate(lines, 1):
        if line < 400:
            continue
        elif line > 2000:
            continue
        elif len(newlines)==1 and len(lines)>=i+2:
            #print("checking", len(newlines), len(lines), i)
            if lines[i]<900:
                continue
        #else:
        #    print(len(newlines), len(lines), i, line, "|", len(newlines)==1,  len(lines)>=i+2)
        #    pass

        #print("<<")
        #print("len(newlines)", len(newlines))
        #print("i", i)
        #print("line", line)
        #print()
        newlines.append(line)
    return newlines








# UNUSED ....

def save_image_with_lines(img, out_fn):
    """helper function to save an image """

    img_lines = img.draw_lines(orig_img_as_background=True)
    print("> saving image with detected lines to '{}'".format(out_fn))
    cv2.imwrite(out_fn, img_lines)

# From pdftabextract:

def generate_hough_lines(lines):
    """
    From a list of lines in <lines> detected by cv2.HoughLines, create a list with a tuple per line
    containing:
    (rho, theta, normalized theta with 0 <= theta_norm < np.pi, DIRECTION_VERTICAL or DIRECTION_HORIZONTAL)
    """
    lines_hough = []
    for l in lines:
        rho, theta = l[0]  # they come like this from OpenCV's hough transform
        theta_norm = normalize_angle(theta)
            
        if abs(PIHLF - theta_norm) > PI4TH:  # vertical
            line_dir = 'v'
        else:
            line_dir = 'h'
        
        lines_hough.append((rho, theta, theta_norm, line_dir))
    
    return lines_hough


def normalize_angle(theta):
    """Normalize an angle theta to theta_norm so that: 0 <= theta_norm < 2 * np.pi"""
    twopi = 2 * np.pi
    
    if theta >= twopi:
        m = math.floor(theta/twopi)
        if theta/twopi - m > 0.99999:   # account for rounding errors
            m += 1        
        theta_norm = theta - m * twopi
    elif theta < 0:
        m = math.ceil(theta/twopi)
        if theta/twopi - m < -0.99999:   # account for rounding errors
            m -= 1
        theta_norm = abs(theta - m * twopi)
    else:
        theta_norm = theta
    
    return theta_norm

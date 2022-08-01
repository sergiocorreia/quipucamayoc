'''
Wrapper around all PDF-related functions
'''

# ---------------------------
# Imports
# ---------------------------

# From stdlib
import sys
import shutil
from pathlib import Path
import collections
import hashlib

##import os
##from lxml import etree

##import pickle


# From pip
import yaml
import rich
from rich.progress import track
from rich.console import Console
from rich.table import Table
from PIL import Image # Pillow!
import cv2


# From this package
##from .images import *
##from .lines import detect_lines_in_page
##from .page import Page

#import ocr_gcv # , ocr_azure, ocr_tesseract
#from .ocr_gcv import run as run_ocr_gcv

##from . import ocr_gcv

from .poppler_wrapper import Poppler
from .utils import *


# ---------------------------
# Constants
# ---------------------------



# ---------------------------
# Functions
# ---------------------------


# ---------------------------
# Class PDF: Manipulate a PDF and save/load to disk
# ---------------------------

class PDF:

    all_engines = ('gcv', 'textract', 'embedded') # 'azure', 'abbyy-online', 'tesseract', 

    def __init__(self, filename,
            cache_folder=None,
            use_cache=False,
            poppler_path=None,
            verbose=False):

        # Input filename
        self.filename = Path(filename)
        if not self.filename.is_file():
            msg = f'PDF file not found: {self.filename}'
            error_and_exit(msg)

        # Set poppler libraries
        self.poppler = Poppler(poppler_path=poppler_path)
        self.poppler.check_binaries()

        # Set working directory (stores temporary and output files)
        if cache_folder is None:
            cache_folder = filename.parent / ('quipu-' + filename.stem[:20])
        else:
            cache_folder = Path(cache_folder)
        if cache_folder.exists() and not cache_folder.is_dir():
            msg = f'Cannot create folder "{cache_folder}" (perhaps location already exists and is a file?)'
            error_and_exit(msg)
        self.cache_folder = cache_folder


        # Load metadata (also sets .first_page and .last_page)
        self.load_info_from_file(verbose)

        # Cache
        self.use_cache = use_cache

        # Create cache_folder
        self.initialize_cache_folder(verbose)


    def load_info_from_file(self, verbose=False):
        if verbose:
            print_update('[PDF] Loading metadata from PDF file')

        # Get info on the PDF
        info = self.poppler.get_info(self.filename)
        self.title = info.get('title', None)
        self.author = info.get('author', None)
        self.creator = info.get('creator', None)
        self.producer = info.get('producer', None)
        self.creation_date = info.get('creation_date', None)
        self.num_pages = info.get('num_pages', None)
        self.size = int(info['size'].split()[0])
        
        self.first_page = 1
        self.last_page = self.num_pages


    def describe(self):
        console = Console()
        table = Table(show_header=True, header_style="bold cyan dim", title='PDF Metadata', style='cyan dim')
        table.add_column("Key", min_width=14, style='cyan')
        table.add_column("Value", justify="left", style='cyan')
        table.add_row('Filename', str(self.filename))
        if self.title:
            table.add_row('Title', self.title)
        if self.author:
            table.add_row('Author', self.author)
        if self.creator:
            table.add_row('Creator', self.creator)
        if self.producer:
            table.add_row('Producer', self.producer)
        if self.creation_date:
            table.add_row('Creation date', self.creation_date)
        table.add_row('Size', f'{self.size / 2 ** 20:<6.2f}MiB')
        table.add_row('Num pages', f'{self.num_pages}')
        table.add_row('Active pages', f'{self.first_page}-{self.last_page} ({self.last_page-self.first_page+1} pages)')
        print()
        console.print(table)
        print()


    def initialize_cache_folder(self, verbose=False):

        if verbose:
            print_update(f'[PDF] Creating working directory: {self.cache_folder}')

        # Create subdirectories

        if not self.cache_folder.is_dir():
            if not self.cache_folder.parent.is_dir():
                print(f'[ERROR] Cannot create cache_folder; parent folder does not exist: {self.cache_folder.parent}')
                sys.exit(1)
            robust_mkdir(self.cache_folder)

        subdirs = ('tmp', 'img_raw', 'img_clean', 'watermarks', 'ocr', 'page_metadata', 'lines', 'done')
        for subdir in subdirs:
            path = self.cache_folder / subdir
            #print('   - (Re)creating folder:', path)
            shutil.rmtree(path, ignore_errors = True)
            robust_mkdir(path)

        for dd in self.all_engines:
            robust_mkdir(self.cache_folder / 'ocr' / dd)

        # GCV folders
        gcv_path = self.cache_folder / 'ocr' / 'gcv'
        robust_mkdir(gcv_path / 'json')  # Raw JSON file is zipped and saved here
        robust_mkdir(gcv_path / 'pickle')  # JSON file converted to Page() object and pickled
        robust_mkdir(gcv_path / 'output')  # tab-separated output
        robust_mkdir(gcv_path / 'debug')  # debugging results
        
        # Save info to YAML
        fn = self.cache_folder / 'info.yaml'
        info = dict()
        info['title'] = self.title
        info['author'] = self.author
        info['creator'] = self.creator
        info['producer'] = self.producer
        info['creation_date'] = self.creation_date
        info['num_pages'] = self.num_pages
        info['size'] = self.size

        info['filename'] = str(self.filename)
        info['first_page'] = self.first_page
        info['last_page'] = self.last_page

        with fn.open('w') as fh:
            yaml.dump(info, fh, default_flow_style=False)
        
        if verbose:
            print_update(f'[PDF] PDF object metadata saved to {fn}')


    @property
    def page_range(self):
        return self.first_page, self.last_page


    @page_range.setter
    def page_range(self, page_range):
        first_page, last_page = page_range
        assert 1 <= first_page <= last_page <= self.num_pages
        self.first_page = first_page
        self.last_page = last_page


    def extract_images(self, first_page=None, last_page=None, verbose=False):
        if first_page is None:
            first_page = self.first_page
        if last_page is None:
            last_page = self.last_page

        # Ensure we don't go out of bounds
        first_page = max(self.first_page, first_page)
        last_page = min(self.last_page, last_page)

        #if verbose:
        #    print(f' - Exporting images from pages {first_page}-{last_page}')
        path = self.cache_folder / 'img_raw' / 'image'
        
        msg = 'Extracting images from pages'
        for page in track(range(first_page, last_page+1), description=msg):
            ## # Skip if cache and final image already exists
            ## if self.cache:
            ##     fn = self.cache_folder / 'img_clean' / f'page_{page:04}.png'
            ##     if fn.is_file():
            ##         continue
            self.poppler.extract_image(self.filename, path, page, verbose=verbose)


    def delete_watermarks(self, verbose=False, debug=False):
        '''PDFs downloaded from Google Books and Hathi Trust often have watermarks

        They complicate the OCR process by creating noise and text we don't want.
        To remove them we use a simple process that works in PDFs with more than one page:
        1) Obtain the size and checksum of every image on every page
        2) If we find duplicates, we consider them watermarks and delete them
        '''

        if verbose:
            print_update('Detecting watermarks...')

        path = self.cache_folder / 'img_raw'
        assert path.is_dir()
        fns = path.glob('image-*')  # pdfimages extracts images with structure "image-<page>-<iter>.ext"
        items = collections.defaultdict(list)
        
        # Collect size and checksum of all images
        for fn in fns:
            size = os.path.getsize(fn)
            checksum = hashlib.md5(fn.open('rb').read()).hexdigest()
            items[(size, checksum)].append(fn)

        # Delete watermark images
        for fns in items.values():
            n = len(fns)
            if n > 1:
                if verbose:
                    print_update(f' - Deleting {n} images with the same size and hash')
                if debug:
                    # Show each watermark once for 1s
                    image = cv2.imread(str(fns[0]))
                    cv2.imshow("Window", image)
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()
                for fn in fns:
                    fn.unlink()

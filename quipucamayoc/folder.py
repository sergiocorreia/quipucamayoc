'''
Internal document class for folder-based books, where images are pages
'''

# ---------------------------
# Imports
# ---------------------------

from pathlib import Path
import datetime

import yaml
from rich.progress import track
from rich.console import Console
from rich.table import Table

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

class Folder:

    all_engines = ('gcv', 'textract')

    def __init__(self, path,
            cache_folder=None,
            use_cache=False,
            verbose=False):

        # Input path
        self.path = Path(path)
        if not self.path.is_dir():
            error_and_exit(f'Folder not found: {self.path}')

        # Set working directory (stores temporary and output files)
        if cache_folder is None:
            cache_folder = path / 'quipucamayoc'
        else:
            cache_folder = Path(cache_folder) / ('quipu-' + path.stem[:20])
        if cache_folder.exists() and not cache_folder.is_dir():
            msg = f'Cannot create folder "{cache_folder}" (perhaps location already exists and is a file?)'
            error_and_exit(msg)
        self.cache_folder = cache_folder

        # Load metadata (also sets .first_page and .last_page)
        self.load_info(verbose)

        # Cache
        self.use_cache = use_cache

        # Create cache_folder
        self.initialize_cache_folder(verbose)


    def load_info(self, verbose=False):
        # TODO: Allow for JSON or other metadata
        if verbose:
            print_update('[PDF] Loading metadata from folder')


        # Get info on the PDF
        self.title = self.path.name
        self.author = ''
        self.creator = ''
        self.producer = ''
        
        creation_date = datetime.datetime.fromtimestamp(self.path.stat().st_mtime, tz=datetime.timezone.utc)
        #self.creation_date = creation_date.isoformat()
        self.creation_date = creation_date.strftime("%d %b %Y %I:%M %p")
        
        fns = self._get_image_list()
        self.num_pages = len(fns)
        self.size = int(sum(fn.stat().st_size for fn in fns))
        self.first_page = 1 if fns else 0
        self.last_page = len(fns) if fns else -1


    def describe(self):
        console = Console()
        table = Table(show_header=True, header_style="bold cyan dim", title='PDF Metadata', style='cyan dim')
        table.add_column("Key", min_width=14, style='cyan')
        table.add_column("Value", justify="left", style='cyan')
        table.add_row('Path', str(self.path))
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
        table.add_row('Cache folder', str(self.cache_folder))
        print()
        console.print(table)
        print()


    def initialize_cache_folder(self, verbose=False):

        if verbose:
            print_update(f'[PDF] Creating working directory: {self.cache_folder}')

        # Create subdirectories
        create_folder(self.cache_folder, exist_ok=True, check_parent=True, try_again=True)

        subdirs = ('tmp', 'img_clean', 'ocr', 'page_metadata', 'lines', 'done')
        for subdir in subdirs:
            create_folder(self.cache_folder / subdir, delete_before=True, try_again=True)

        for dd in self.all_engines:
            create_folder(self.cache_folder / 'ocr' / dd, try_again=True)

        # GCV folders
        gcv_path = self.cache_folder / 'ocr' / 'gcv'
        create_folder(gcv_path / 'json', try_again=True)  # Raw JSON file is zipped and saved here
        create_folder(gcv_path / 'pickle', try_again=True)  # JSON file converted to Page() object and pickled
        create_folder(gcv_path / 'output', try_again=True)  # tab-separated output
        create_folder(gcv_path / 'debug', try_again=True)  # debugging results
        
        # Save info to YAML
        fn = self.cache_folder / 'info.yaml'
        info = dict()
        info['title'] = self.title
        #info['author'] = self.author
        #info['creator'] = self.creator
        #info['producer'] = self.producer
        info['creation_date'] = self.creation_date
        info['num_pages'] = self.num_pages
        info['size'] = self.size

        info['path'] = str(self.path)
        info['first_page'] = self.first_page
        info['last_page'] = self.last_page

        with fn.open('w') as fh:
            yaml.dump(info, fh, default_flow_style=False)
        
        if verbose:
            print_update(f'[PDF] Folder object metadata saved to {fn}')


    @property
    def page_range(self):
        return self.first_page, self.last_page


    @page_range.setter
    def page_range(self, page_range):
        first_page, last_page = page_range
        assert 1 <= first_page <= last_page <= self.num_pages
        self.first_page = first_page
        self.last_page = last_page


    def _get_image_list(self):
        extensions = ('.bmp', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.webp', '.tiff', '.tif')
        fns = sorted(p.resolve() for p in Path(self.path).glob("*") if p.suffix in extensions)
        return fns


    def extract_images(self, first_page=None, last_page=None, verbose=False):
        '''Just copy+rename images to the 'img_clean' folder as that's what subsequent code expects'''

        if first_page is None:
            first_page = self.first_page
        if last_page is None:
            last_page = self.last_page

        # Ensure we don't go out of bounds
        first_page = max(self.first_page, first_page)
        last_page = min(self.last_page, last_page)

        output_path = self.cache_folder / 'img_clean'
        
        # Get list of images
        fns = self._get_image_list()

        if verbose:
            print_update(f' - {len(fns)} images detected')

        msg = ' [green]- Copying images to cache     '
        for page, fn in enumerate(track(fns, description=msg), 1):
            new_fn = output_path / f'page-{page:04}{fn.suffix}'
            shutil.copyfile(fn, new_fn)


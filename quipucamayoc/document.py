'''
Document class that combines PDF and image-based documents

Quipucamayoc supports two types of documents:

1. A single PDF with multiple pages
2. A folder with images, each representing a single page

The 'Document' class wraps these two types of objects
'''


# ---------------------------
# Imports
# ---------------------------

from pathlib import Path
import sys

import rich

from .pdf import PDF


# ---------------------------
# Main class
# ---------------------------

class Document:

    def __init__(self, filename,
            cache_folder=None,
            use_cache=False,
            poppler_path=None,
            verbose=False):

        self.source = Path(filename)
        self.is_pdf = self.source.suffix == '.pdf'
        if not self.is_pdf and not self.source.is_dir():
            rich.print(f'[ERROR] Expected a PDF filename or a folder, but received: {self.source}')
            sys.exit(1)

        if self.is_pdf:
            self.core = PDF(self.source, cache_folder, use_cache, poppler_path, verbose)
        else:
            raise NotImplementedError


    def describe(self):
        self.core.describe()


    def extract_images(self, first_page=None, last_page=None, verbose=False):
        self.core.extract_images(first_page, last_page, verbose)


    def delete_watermarks(self, verbose=False, debug=False):
        if self.is_pdf:
            self.core.delete_watermarks(verbose, debug)
        else:
            pass  # No watermarks in pictures


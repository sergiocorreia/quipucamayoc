'''
Poppler interface/wrapper

We use this because there might be different ways of accessing poppler, i.e.:

- Installed and accessible through PATH
- Installed and provided through a --poppler-dir or --poppler-directory option
- Via python-poppler; not available on Windows
- Using included binaries (.exe and .dll); only on Windows
'''

# ---------------------------
# Imports
# ---------------------------

import sys
from pathlib import Path
import importlib.resources
import platform

from .utils import *


# ---------------------------
# Main class
# ---------------------------

class Poppler:

    def __init__(self, poppler_path=None):

        # Decide which path to use:
        # 1) Folder given by user
        # 2) Folder included with the quipucamayoc package (on Windows)
        # 3) System search path (whatever is installed and available)

        use_system = False

        if poppler_path:
            self.poppler_path = Path(poppler_path)
        elif platform.system()=='Windows':
            self.poppler_path = Path(importlib.resources.files(__package__)) / 'poppler-windows'
        else:
            use_system = True

        f = lambda x : x if use_system else (self.poppler_path / x).as_posix()
        self.pdfinfo = f('pdfinfo.exe')
        self.pdfimages = f('pdfimages.exe')
        self.pdftohtml = f('pdftohtml.exe')
        self.pdftotext = f('pdftotext.exe')


    def check_binaries(self, verbose=False):
        '''Check that the poppler binaries exist'''

        binaries = ('pdfinfo', 'pdfinfo', 'pdfimages', 'pdftohtml', 'pdftotext')
        for binary in binaries:
            if verbose:
                print_update(f'==== Checking -{binary}- from POPPLER ====')
                cmd = self.__dict__[binary]
                c = Command(cmd, args='-v', ignore_error=True)
                if c.exitcode != 0:
                    #print(c.err)
                    print(c.err.strip() + '\n')
                    error_and_exit(f'Required poppler executable {binary} not found')
                assert c.out == ''
                assert c.err != ''


    def get_info(self, filename):
        c = Command(self.pdfinfo, args=str(filename))
        out = c.out.replace('\r', '').split('\n')

        # Replace some keys
        keys = {'creationdate': 'creation_date', 'file_size': 'size', 'pages': 'num_pages'}
        
        info = {}
        for row in out:
            if not row: continue
            k, v = row.split(':', 1)
            k = k.lower().replace(' ', '_').strip()
            v = v.strip()
            k = keys.get(k, k)
            vv = v
            if (v.lower() == 'no'): vv = False
            if (v.lower() == 'yes'): vv = True
            if (v.lower() == 'none'): vv = None
            if (v.isdigit()): vv = int(v)
            if vv is not None:
                info[k] = vv
        return info


    def extract_image(self, filename, path, page, verbose=False):
        #args = ['-p', '-png', '-f', str(page), '-l', str(page)]
        args = ['-p', '-all', '-f', str(page), '-l', str(page)]
        args.extend([filename, path])
        #if verbose:
        #    print('   [CMD]', self.pdfimages, ' '.join(str(x) for x in args))
        c = Command(self.pdfimages, args)


    def get_page_info(self, filename, path, page, verbose=False):
        args = ['-xml', '-nomerge', '-zoom', '1', '-f', str(page), '-l', str(page), filename, path]
        # we need "-zoom 1" else it defaults to 1.5 and messes up the unit conversion
        c = Command(PDFTOHTML, args)

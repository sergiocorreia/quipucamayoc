# ---------------------------
# Imports
# ---------------------------

import os
import sys
import shutil
#import csv
#import time
import shlex
#import bisect
#import unicodedata

#from pathlib import Path
import pathlib
from subprocess import Popen, PIPE

from rich.console import Console


# ---------------------------
# Constants
# ---------------------------

console = Console()


# ---------------------------
# Utility functions
# ---------------------------

def error_and_exit(message):
    message = '[bold]ERROR: [/]' + message
    console.print(message, style='red')
    sys.exit(1)


def print_update(message):
    console.print(message, style='green')


def robust_mkdir(name):
    try:
        name.mkdir()
    except PermissionError:
        print(' - Sleeping 1s due to PermissionError')
        time.sleep(1)
        name.mkdir()


def delete_and_create_folder(path):
    shutil.rmtree(path, ignore_errors = True) # Delete in case it already exists
    debug_path.mkdir()
    return path


def create_folder(path, exist_ok=False, check_parent=False, delete_before=False, try_again=False):
    assert isinstance(path, pathlib.PurePath)

    assert not (exist_ok and delete_before), 'Cannot ask to delete beforehand and allow for preexistence'

    if check_parent and not path.parent.is_dir():
        error_and_exit(f'Folder {path.parent} does not exist; cannot create {path.name}')

    if delete_before:
        shutil.rmtree(path, ignore_errors=True)

    if try_again:
        try:
            path.mkdir(exist_ok=exist_ok)
        except PermissionError:
            print(' - Sleeping 1s due to PermissionError')
            time.sleep(1)
            path.mkdir(exist_ok=exist_ok)
    else:
        path.mkdir(exist_ok=exist_ok)
    
    return path  # Also returns the path so you can do: new_path = create_folder(basepath / 'foo')


# ---------------------------
# Utility classes
# ---------------------------


class Command(object):

    def __init__(self, cmd, args=None,
                 ignore_error=False, cwd=None,
                 verbose=False):
        self.cmd = cmd
        self.args = [] if args is None else args
        self.out = None
        self.err = None
        self.exitcode = None
        self.cwd = cwd
        self.run(ignore_error=ignore_error, verbose=verbose)

    def run(self, msg='', ignore_error=False, verbose=False):

        # Fix Windows error if passed a string
        if isinstance(self.args, str):
            self.args = shlex.split(self.args, posix=(os.name != "nt"))
            if os.name == "nt":
                self.args = [arg.replace('/', '\\') for arg in self.args]
        else:
            # With this we can pass pathlib objects
            self.args = [arg if isinstance(arg, str) else str(arg) for arg in self.args]

        if verbose:
            print(self.cmd, *self.args)

        proc = Popen([self.cmd] + self.args, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True, cwd=self.cwd)
        self.out, self.err = proc.communicate(input=msg.encode('utf-8'))
        self.out = self.out.decode('utf-8')
        self.err = self.err.decode('utf-8')
        self.exitcode = proc.returncode
        
        if not ignore_error and self.exitcode:
            raise IOError(self.err)

    def inspect(self):
        print('cmd:', self.cmd)
        print('args:', self.args)
        print('exit code:', self.exitcode)
        print('out:', self.out)
        print('err:', self.err)


'''
CLI providing some of the main tools for non-python users
'''


# ---------------------------
# Imports
# ---------------------------

#import os
#import os.path as p
from pathlib import Path
import sys
import errno
import os

import click

from .aws_extract_tables import aws_extract_tables
from .aws_setup import install_aws
from .aws_setup import uninstall_aws


# ---------------------------
# Functions
# ---------------------------



# ---------------------------
# Main
# ---------------------------

@click.group(help='This help file lists the commandline commands available for quipucamayoc')
def cli():
    pass

@cli.command(help='Install and uninstall AWS support')  # @cli, not @click!
@click.argument('action', type=click.Choice(['install', 'uninstall']))
def aws(action):
    if action == 'install':
        install_aws()
    elif action == 'uninstall':
        uninstall_aws()
    else:
        raise Exception


@cli.command(help='Apply OCR and extract tables using AWS')
@click.option('-f', '--filename', '--file', type=str, help='Filename of file that will be OCRed')
@click.option('-d', '--directory', '--dir', type=str, help='Alternatively, folder containing images that will be OCRed')
@click.option('-e', '--extension', '--ext', type=str, help='Extension of images (pdf, png, jpg, jpeg)')
@click.option('--keep/--no-keep', default=False, help='Keep object in AWS S3 bucket (will have to be removed manually)')
@click.option('--engine', default='aws', type=click.Choice(['aws']), help='OCR engine (currently only AWS)')
@click.option('-o','--outputformat', '--out', type=str, default="tsv", help="Format of output (csv, tsv, ...)")
@click.option("--page-append", default=False, is_flag=True, help="Append all tables on page and pages in file to single output.")
@click.option("--output-dir", type=str, default=None, help="Name an (existing or not) directory for output contents to be placed into")
def extract_tables(filename, directory, extension, keep, engine, outputformat, page_append, output_dir):
    if engine == 'aws':
        if filename is not None:
                filename = Path(filename)
                if not filename.is_file():
                    raise SystemExit(f"[quipucamayoc] File not found: '{filename.name}'")
        directory = Path(directory) if directory is not None else directory
        aws_extract_tables(filename=filename, 
                           directory=directory, 
                           extension=extension, 
                           keep_in_s3=keep, 
                           ignore_cache=True, 
                           output=outputformat, 
                           page_append=page_append,
                           output_dir=output_dir)
    else:
        raise Exception(engine)

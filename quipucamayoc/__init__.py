"""
Quipucamayoc: tools for digitizing historical data
====================================

Quipucamayoc is a Python package that provides
a set of building blocks to implement large-scale
digitization of historical structured data,
such as balance sheets or price lists.
"""

from .version import __version__
from .cli import cli
from .aws_extract_tables import aws_extract_tables
from .aws_setup import install_aws
from .aws_setup import uninstall_aws

from .document import Document
#from .pdf import PDF
from .poppler_wrapper import Poppler

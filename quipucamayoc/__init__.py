"""
Quipucamayoc: tools for digitizing historical data
====================================

Quipucamayoc is a Python package that provides
a set of building blocks to implement large-scale
digitization of historical structured data,
such as balance sheets or price lists.
"""

from .version import __version__
from .aws_extract_tables import aws_extract_tables
from .aws_setup import install_aws
from .aws_setup import uninstall_aws


def extract_tables(filename=None, directory=None, extension=None, engine='aws'):
	assert engine in ('aws',)
	if engine == 'aws':
		aws_extract_tables(filename, directory, extension)
	else:
		raise Exception
	print('DONE!')

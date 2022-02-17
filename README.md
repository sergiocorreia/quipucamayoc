# Quipucamayoc: tools for digitizing historical data

[![Development Status](https://img.shields.io/pypi/status/quipucamayoc.svg)](https://pypi.python.org/pypi/quipucamayoc/)
[![Build Status](https://github.com/sergiocorreia/quipucamayoc/workflows/CI%20Tests/badge.svg)](https://github.com/sergiocorreia/quipucamayoc/actions?query=workflow%3A%22CI+Tests%22)
![License](https://img.shields.io/pypi/l/quipucamayoc.svg)
[![DOI](https://zenodo.org/badge/55024750.svg)](https://zenodo.org/badge/latestdoi/55024750)

[![GitHub Releases](https://img.shields.io/github/tag/sergiocorreia/quipucamayoc.svg?label=github+release)](https://github.com/sergiocorreia/quipucamayoc/releases)
[![Python version](https://img.shields.io/pypi/pyversions/quipucamayoc.svg)](https://pypi.python.org/pypi/quipucamayoc/)
[![Supported implementations](https://img.shields.io/pypi/implementation/quipucamayoc.svg)](https://pypi.org/project/quipucamayoc)

[quipucamayoc](http://scorreia.com/software/quipucamayoc/) is a Python package that simplifies the extraction of historical data from scanned images and PDFs.
It's designed to be modular and so it can be used together with other existing tools, and can be extended easily by users.

For an overview of how to use  `quipucamayoc` to digitize historical data, see [this research article](http://scorreia.com/research/digitizing.pdf), which amongst other things details the different steps involved, the methods used, and provides practical examples.
For an user guide, documentation, and installation instructions, see <http://scorreia.com/software/quipucamayoc/> (TODO).

If you want to contribute by improving the code or extending its functionality (much welcome!), head [here](/CONTRIBUTING.md).


## Installation

### Pip

To manage quipucamayoc using pip, open the command line and run:

- `pip install quipucamayoc` to install
    - `pip install "quipucamayoc[dev]"` to include extra dependencies used when developing the code
- `pip install -U quipucamayoc` to upgrade
- `pip uninstall quipucamayoc` to remove

Note that `quipucamayoc` has been tested against Python 3.10 and newer versions, but should also work with Python 3.9.

### Git Install

After cloning the repo to your computer and navigating to the quipucamayoc folder, run:

- `python setup.py install` to install the package locally
- `python setup.py develop` to install locally with a symlink so changes are automatically updated (recommended for developers)


## Contributing

Feel free to submit push requests. For consistency, code should comply with [pep8](https://pypi.python.org/pypi/pep8) (as long as its reasonable), and with the style guides by [@kennethreitz](http://docs.python-guide.org/en/latest/writing/style/) and [google](http://google.github.io/styleguide/pyguide.html). Read more [here](/CONTRIBUTING.md).


## Citation

[(Download BibTex file here)](https://raw.githubusercontent.com/sergiocorreia/quipucamayoc/master/quipucamayoc.bib)

#### As text

<ul>
<li>
Sergio Correia, Stephan Luck: “Digitizing Historical Balance Sheet Data: A Practitioner's Guide”, 2022; <a href='http://arxiv.org/abs/1903.01633'>arXiv:1903.01633</a>.
</li>
</ul>

TODO: POST NEW VERSION OF PAPER IN ARXIV AND CHANGE ARXIV LINK. ALSO UPDATE BOTH BIBTEX LINKS!

#### As BibTex

```bibtex
@misc{quipucamayoc,
  Author = {Correia, Sergio and Luck, Stephan},
  Title = {Digitizing Historical Balance Sheet Data: A Practitioner's Guide},
  Year = {2022},
  eprint = {arXiv:1903.01633},
  journal={arXiv preprint arXiv:1903.01633}
}
```

## Acknowledgments

Quipucamayoc is built upon the work and improvements of many users and developers, from which it was heavily inspired, such as:

- [pdftabextract](https://github.com/WZBSocialScienceCenter/pdftabextract)

It is also relies for most of its work on the following open source projects:

- [Python](https://www.python.org/)
- [NumPy](https://numpy.org/)
- [OpenCV](https://opencv.org/) and [OpenCV-Python](https://github.com/opencv/opencv-python)
- [Poppler](https://poppler.freedesktop.org/)


## License

Quipucamayoc is developed under the [GNU Affero GPL v3 license](https://www.gnu.org/licenses/agpl-3.0.en.html).


## Why "quipucamayoc"?

The _quipucamayocs_ were the Inca empire officials in charge of desciphering (amonst other things) accounting information stored in quipus. Our goal for this package is to act as a sort of quipucamayoc, helping researchers in desciphering and extracting historical information, particularly balance sheets and numerical records.

<p align="center">
  <a href="https://en.wikipedia.org/wiki/Quipu" rel="quipu"><img src="https://github.com/sergiocorreia/quipucamayoc2/blob/master/docs/quipucamayoc.png?raw=true" /></a>
</p>

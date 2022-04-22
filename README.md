# Quipucamayoc: tools for digitizing historical data
This is an incomplete version, forked from the complete version, to make specific changes.
 is a Python package that simplifies the extraction of historical data from scanned images and PDFs.
It's designed to be modular and so it can be used together with other existing tools, and can be extended easily by users.

For an overview of how to use  `quipucamayoc` to digitize historical data, see [the original quipucamayoc repository, here][https://github.com/sergiocorreia/quipucamayoc]


## Version Notes
To use this version, you must follow the installation notes for git below. This version is not available through pypi.
This version has several added features under development.
Notably, 
- output can now be specified with -o to output a csv file, in addition to a tsv as default.
- Output can also be turned to a single file using --page-append , rather than to a file per table found. For files with multi-page tables (especially where all tables in the file have the same construction), this is remarkably useful. 
- Timeouts are now fully caught, where previously upon timeout it could simply pass as though it had succeded

Current cautions:
- Currently, when appending new tables to an existing table (using --page-append), the top 2 rows are removed. The data I have been primarily testing has a two-line header, however this may be made more modular in due time. I wonder if some sort of quipu defaults file should or could exist to record this, like environment variables.
- -d (with --extension pdf) is currently under development. Results may vary. 
  Each file is processed as an individual file outputted to a common directory, 
  with a .done file. When the entire dir is processed, all .done files are replaced
  with a single directory .done, which contains the names of all removed .done files.
  Does not currently leverage asyncrinosity, so can probably be made MUCH faster soon.


## Installation



Note that `quipucamayoc` has been tested against Python 3.10 and newer versions, but should also work with Python 3.9.

### Git Install

After cloning the repo to your computer and navigating to the quipucamayoc folder, run:

- `pip install .` to install the package locally
- `pip install -e .` to install locally with a symlink so changes are automatically updated (recommended for developers)


## After installation

### AWS

AWS configuration is quite cumbersome, so it has been automated. To do so, follow these four steps:

1. [Download](https://aws.amazon.com/cli/) and install the `aws` command line interface (CLI).  *Update: `quipucamayoc` installs the `awscli` package so this step might not be necessary anymore*.
2. [Configure](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html) your credentials with `aws configure`. This requires an Amazon/AWS account.
3. From the command line, run the quipucamayoc command `quipu aws install`


Notes:

- You can avoid step 1 by directly [writing your credentials[(https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)] to the `credentials` file.
- Steps 3-4 are also available from within Python in the `setup_textract()` and `test_textract()` functions.
- If you want to remove all quipucamayoc artifacts from your AWS account, you can run `quipu aws uninstall` from the command line.
- The default [AWS region](https://www.concurrencylabs.com/blog/choose-your-aws-region-wisely/) is `aws-east-1`. To use other regions, use the `--region <name>` option.


## Usage

- From the command line, you can extract tables using AWS via `quipu extract-tables --filename <myfile.pdf>`


## TODO

- [x] Automatically set up Textract pipeline
- [ ] Expose key functions as command line tools
- [ ] Allow parallel (async?) tasks. Useful for OpenCV (CPU-intensive) and Textract calls (IO-intensive). Consider also [uvloop](https://github.com/MagicStack/uvloop)
- [ ] Include Poppler by default on Windows
- [ ] Add mypy/(flake8|black)


## Contributing

Feel free to submit push requests. For consistency, code should comply with [pep8](https://pypi.python.org/pypi/pep8) (as long as its reasonable), and with the style guides by [@kennethreitz](http://docs.python-guide.org/en/latest/writing/style/) and [google](http://google.github.io/styleguide/pyguide.html). Read more [here](/CONTRIBUTING.md).


## Citation

[(Download BibTex file here)](https://raw.githubusercontent.com/sergiocorreia/quipucamayoc/master/quipucamayoc.bib)

#### As text

<ul>
<li>
Sergio Correia, Stephan Luck: “Digitizing Historical Balance Sheet Data: A Practitioner's Guide”, 2022; <a href='http://arxiv.org/abs/2204.00052'>arXiv:2204.00052</a>.
</li>
</ul>


#### As BibTex

```bibtex
@misc{quipucamayoc,
  Author = {Correia, Sergio and Luck, Stephan},
  Title = {Digitizing Historical Balance Sheet Data: A Practitioner's Guide},
  Year = {2022},
  eprint = {arXiv:2204.00052},
  journal={arXiv preprint arXiv:2204.00052}
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

# About

This folder contains a slightly modified version of DocTr, downloaded from:
https://github.com/fh2019ustc/DocTr

See also the included LICENSE file.


# Why?

Because DocTr is not available as a package, we modified `inference.py`
to be easier to access for Quipucamayoc, and included all related python files
(i.e., not the files used for training).

Minor modifications also include:

- Rename `inference.py` into `__init__.py`
- Allow CPU usage, by replacing .cuda() instances with conditional calls
- Make relative packages explicit by adding a dot (e.g. "from .seg ...")


# Installing

To use this method you need to:

1) Install PyTorch, following the instructions available at https://pytorch.org/

2) Download the two pre-trained models linked from https://github.com/fh2019ustc/DocTr . As of 02Aug2022, the link is:

https://drive.google.com/drive/folders/1eZRxnRVpf5iy3VJakJNTKWw5Zk9g-F_0

These three models should be unzipped and placed in an accessible place, so their name can be passed to the .dewarp(method='DocTr') function

3) Install timm: https://pypi.org/project/timm/

# CLI execution

We also included the unmodified `inference.py`, which can be run as follows:

(NOT WORKING; TODO)

```bash
python infer.py --wc_model_path C:/MyModel/unetnc_doc3d.pkl --bm_model_path C:/MyModel/dnetccnl_doc3d.pkl --show --img_path C:/WarpedImages --out_path C:/CleanImages
```

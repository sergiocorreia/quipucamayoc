# About

This folder contains a slightly modified version of DewarpNet, downloaded from:
https://github.com/cvlab-stonybrook/DewarpNet

See also the included LICENSE file.


# Why?

Because DewarpNet is not available as a package, we modified `infer.py`
to be easier to access for Quipucamayoc, and included all related python files
(i.e., not the files used for training).

Minor modifications also include:

- Remove PyTorch warning by adding align_corners=False to grid_sample() call
- Make relative packages explicit by adding a dot (e.g. "from .models ...")
- Removed unused imports (tqdm, matplotlib, json)

# Installing

To use this method you need to:

1) Install PyTorch, following the instructions available at https://pytorch.org/

2) Download the two pre-trained models linked from https://github.com/cvlab-stonybrook/DewarpNet . As of 02Aug2022, the link is:

https://drive.google.com/file/d/1hJKCb4eF1AJih_dhZOJSF5VR-ZtRNaap/view?usp=sharing

These two models should be unzipped and placed in an accessible place, so their name can be passed to the .dewarp(method='dewarpnet') function


# CLI execution

We also included the unmodified `infer.py`, which can be run as follows:

```bash
python infer.py --wc_model_path C:/MyModel/unetnc_doc3d.pkl --bm_model_path C:/MyModel/dnetccnl_doc3d.pkl --show --img_path C:/WarpedImages --out_path C:/CleanImages
```

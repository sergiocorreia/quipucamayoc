## Generating distribution archives

First, ensure build and twine are installed:

```
py -m pip install --upgrade build
py -m pip install --upgrade twine
```

Then, to build:

```
py -m build
```

To test for misc. errors:


```
pip install readme_renderer[md]
py -m build
twine check dist/*
```


To test uploading:

```
py -m twine upload --repository testpypi dist/*
```


To upload to Pypi:

```
py -m twine upload --repository pypi dist/*
```


## Miscellaneous

- The list of possible classifiers for `setup.cfg` is here: https://pypi.org/pypi?%3Aaction=list_classifiers
- Ensure dependencies are upgraded (twine, etc.): https://packaging.python.org/en/latest/tutorials/packaging-projects/#uploading-the-distribution-archives


- Example .cfg files are:
	- https://github.com/pallets/click/blob/main/setup.cfg
	- https://github.com/pandas-dev/pandas/blob/main/setup.cfg

For help on distributing extra files on install, see:

- https://jwodder.github.io/kbits/posts/pypkg-data/

Add to pypi

Upload to arxiv
Get preprint number and add that to BIB



``` python
# https://jwodder.github.io/kbits/posts/pypkg-data/
import importlib.resources as importlib_resources
pkg = importlib_resources.files('quipucamayoc')
pkg_data_file = pkg / "data" / "data.csv"
with pkg_data_file.open() as fp:
    # Do things with fp

# To get the path to the file, call importlib_resources.as_file() on it and use the return value as a context manager:
with importlib_resources.as_file(pkg_data_file) as path:
    # Do things with the pathlib.Path object that is `path`
```

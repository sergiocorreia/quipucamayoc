# Poppler notes

`quipucamayoc` relies on [Poppler](https://poppler.freedesktop.org/) for some of its PDF handling tools, such as:

1. Extracting images from PDFs
2. Extracting text from PDFs (usually embedded text from PDfs already OCRed)

Sadly as of 2022 there are no [multi-platform Python bindings](https://github.com/cbrunet/python-poppler/issues/9) so we rely on calling the Poppler tools as command line utilities. Even then, installing Poppler on Windows might be a bit confusing so below we describe the alternative steps:

1. Install the latest release from the [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases) Github repo.
2. Alternatively, install Poppler on Windows using Anaconda
3. Experimentally, `quipucamayoc` includes Poppler, although it might not correspond to the latest available release.

Note that for simplicity we include all files from `poppler-windows` (37 items; 20mb uncompressed) but only a subset is needed (18 files, 15mb):

- freetype.dll
- jbig.dll
- lcms2.dll
- Lerc.dll
- libcrypto-3-x64.dll
- libcurl.dll
- libdeflate.dll
- libpng16.dll
- libssh2.dll
- openjp2.dll
- pdfimages.exe
- pdfinfo.exe
- pdftohtml.exe
- pdftotext.exe
- poppler.dll
- tiff.dll
- zlib.dll
- zstd.dll


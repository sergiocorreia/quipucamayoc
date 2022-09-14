"""
Quick `quipucamayoc` demo

You might also need:
 pip install jupyterlab
 pip install nodejs
 jupyter-lab


API guide:

- The two main objects are the Document and Page objects, which abstract away all the code dealing with images and/or PDFs.
- Most methods have a `verbose` argument which will show detailed information when set to True
- Some methods also have a `debug` argument which will save intermediate files (in the cache folder) when set to True
- All main objects should have a describe() method, including Document and Page

Document class methods and attributes:
 - source: input file or folder
 - is_pdf
 - cache_folder
 - status (True after cleanup_image is run)
 - models: ML models already loaded to memory
 - extract_images(): will populate the `img_raw` cache folder 
 - cleanup_images(): will delete watermarks and combine images that belong to a single page; saving image in `img_clean`
 - NOTE: cleanup_images() is only necessary for PDF inputs.
 - initialize_pages(): Create a `pages` method with the list of page objects

Page class:
 - load()
 - unload()
 - save()
 - view()
 - rotate()
 - remove_black_background()
 - remove_fore_edges()
 - dewarp()
 - describe()
 - convert()  maybe just convert_grayscale()
 - equalize() INSTEAD OF apply_clahe()
 - binarize()
XXXXXXXXXXXXXXXXXX
 - run_ocr(engine='aws', extract_tables=True) ==> create attributes as in paper
 - detect_lines(...)
 - Maybe a DL example as in page 19


Convenience functions/classes:
- Spellchecker
- Template process_data


TODO:
- Save self.status to cache folder
- More generally, allow LOADING from cache folder (but for that its best to have a stable API)
- Autocreate documentation website (sphinx?)
- Interactive notebook
"""


# ---------------------------
# Imports
# ---------------------------

from pathlib import Path
import quipucamayoc as q


# ---------------------------
# Setup
# ---------------------------

print(q.__version__)
q.Poppler().check_binaries()  # validate the existence of Poppler binaries (optional; also done internally later)

# Constants
input_path = Path('C:/Git/quipucamayoc2/docs/demo/organization-reports')  # either a) folder where we keep the images, or b) a single PDF.
cache_path = Path('C:/WH/quipu')  # where we store intermediate results and images

# Create `document` object
doc = q.Document(input_path, cache_folder=cache_path, verbose=True)
doc.describe()
doc.extract_images(first_page=1, last_page=3, verbose=True)
doc.initialize_pages(verbose=True)


# ---------------------------
# WIP
# ---------------------------

pagenum = 0

# Remove fore-edges
page = doc.pages[pagenum]
page.load()
page.describe()
page.view()
page.convert_to_grayscale()
page.save()
#page.equalize(method='clahe', verbose=True, debug=True, clip_limit=2)
#page.binarize(method='adaptive', verbose=True, debug=True)
#page.binarize(method='sauvola', verbose=True, debug=True)
#page.binarize(method='wolf', verbose=True, debug=True, k=0.2, block_size=5)



print('Done!')
exit()


# ---------------------------
# Process a single page
# ---------------------------

pagenum = 0

# Remove fore-edges
page = doc.pages[pagenum]
page.load()
page.remove_fore_edges(threshold_parameter=188, verbose=True, debug=True)
page.save(debug_name='remove-fore-edges.jpg', verbose=True)
page.unload()

# Remove black backgrounds
page = doc.pages[pagenum]
page.load()
page.remove_black_background(verbose=True, debug=True)
page.save(debug_name='remove-black-background.jpg', verbose=True)
page.unload()


exit()

# Simple dewarping (2D page dewarping AKA perspective transform)
page = doc.pages[pagenum]
page.load()
page.dewarp(method='simple', verbose=True)
page.save(debug_name='dewarp-simple.jpg', verbose=True)
page.unload()


# Deep learning extension: DewarpNet (2019)
model_path = Path('C:/WH/models/DewarpNet')
page = doc.pages[pagenum]
page.load()
page.dewarp(method='DewarpNet', model_path=model_path, verbose=True)
page.save(debug_name='dewarp-DewarpNet.jpg', verbose=True)
page.unload()


# Deep learning extension: DocTr (2021)
model_path = Path('C:/WH/models/DocTr')
page = doc.pages[pagenum]
page.load()
page.dewarp(method='DocTr', model_path=model_path, rectify_illumination=False, verbose=True)
page.save(debug_name='dewarp-DocTr1.jpg', verbose=True)
page.unload()


# Deep learning extension: DocTr with illumination (2021)
# Speed warning: the illumination step might get stuck on complicated inputs on CPU
model_path = Path('C:/WH/models/DocTr')
page = doc.pages[pagenum]
page.load()
page.dewarp(method='DocTr', model_path=model_path, rectify_illumination=True, verbose=True)
page.save(debug_name='dewarp-DocTr2.jpg', verbose=True)
page.unload()






exit()

model_path = Path('C:/WH/dewarpnet')
for page in doc.pages:
	page.load()
	#page.view()
	#page.dewarp(verbose=True, debug=True)
	page.dewarp(method='dewarpnet', model_path=model_path) #, verbose=True, debug=True)
	page.save(verbose=True)
	page.unload()

exit()





# Dewarp demo
fn = Path('C:/Git/quipucamayoc2/docs/demo/fraser-call-report-1941.pdf')
doc = q.Document(fn, cache_folder='C:/WH/quipu', verbose=True)
doc.describe()
doc.extract_images(first_page=1, last_page=1, verbose=True)
doc.cleanup_images(verbose=True, debug=False)
doc.initialize_pages(verbose=True)
page = doc.pages[0]
page.load()
page.dewarp(verbose=True, debug=True)

exit()





fn = Path('C:/Git/quipucamayoc2/docs/demo/snippet-1883.pdf')
doc = q.Document(fn, cache_folder='C:/WH/quipu', verbose=True)

doc.describe()
doc.extract_images(first_page=1, last_page=3, verbose=True)
doc.cleanup_images(verbose=True, debug=False)
doc.initialize_pages(verbose=True)

page = doc.pages[0]
#print(page.filename)
#print(page.pagenum)

page.load()
#page.view(wait=1000)
page.remove_black_background(verbose=True, debug=True)
#page.view(wait=1000)





#from PIL import Image
#with Image.open("C:/WH/borrar/-001-000.jp2") as im:
#    im.rotate(45).show()

print('\nDone!')
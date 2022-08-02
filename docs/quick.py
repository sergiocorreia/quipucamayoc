from pathlib import Path
import quipucamayoc as q

#print(q.__version__)

pop = q.Poppler()
pop.check_binaries()


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

page.remove_fore_edges()



#from PIL import Image
#with Image.open("C:/WH/borrar/-001-000.jp2") as im:
#    im.rotate(45).show()

print('\nDone!')
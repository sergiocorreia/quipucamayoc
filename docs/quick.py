from pathlib import Path
import quipucamayoc as q

#print(q.__version__)

pop = q.Poppler()
pop.check_binaries()





# Folder demo
path = Path('C:/Git/quipucamayoc2/docs/demo/organization-reports')
doc = q.Document(path, cache_folder='C:/WH/quipu', verbose=True)
doc.describe()
doc.extract_images(first_page=1, last_page=3, verbose=True)
doc.initialize_pages(verbose=True)

wc_model_path = Path('C:/WH/dewarpnet/unetnc_doc3d.pkl')
bm_model_path = Path('C:/WH/dewarpnet/dnetccnl_doc3d.pkl')
for page in doc.pages:
	page.load()
	#page.view()
	#page.dewarp(verbose=True, debug=True)
	page.dewarp(method='dewarpnet', wc_model_path=wc_model_path, bm_model_path=bm_model_path) #, verbose=True, debug=True)
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

page.remove_fore_edges(verbose=True, debug=True, threshold_parameter=188)



#from PIL import Image
#with Image.open("C:/WH/borrar/-001-000.jp2") as im:
#    im.rotate(45).show()

print('\nDone!')
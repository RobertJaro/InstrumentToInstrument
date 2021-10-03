import glob
import os

from skimage.io import imsave

from iti.translate import KSOFilmToCCD

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

base_path = "/gss/r.jarolim/iti/film_v8"
prediction_path = os.path.join(base_path, 'compare')
os.makedirs(prediction_path, exist_ok=True)
# create translator
translator = KSOFilmToCCD(resolution=512, model_path=os.path.join(base_path, 'generator_AB.pt'))

# load maps
map_files = sorted(list(glob.glob('/gss/r.jarolim/data/filtered_kso_plate/*.fts.gz', recursive=True)))

# translate
for s_map, kso_img, iti_img in translator.translate(map_files):
    imsave(os.path.join(prediction_path, '%s_film.jpg' % s_map.date.datetime.isoformat('T')), kso_img[0],
           check_contrast=False)
    imsave(os.path.join(prediction_path, '%s_reconstruction.jpg' % s_map.date.datetime.isoformat('T')), iti_img[0],
           check_contrast=False)

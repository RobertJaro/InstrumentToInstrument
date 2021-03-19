import glob
import os
import shutil

img_path = '/gss/r.jarolim/data/converted/kso_synoptic_img'
data_path = "/gss/r.jarolim/data/kso_synoptic"
# copy
data_path_lq = "/gss/r.jarolim/data/kso_synoptic_lq"
os.makedirs(data_path_lq)

base_names = [os.path.basename(f).replace('.jpg', '.fts.gz') for f in glob.glob(img_path + '/*.jpg')]
remove_paths = [f for f in glob.glob(data_path + '/*.fts.gz') if os.path.basename(f) not in base_names]

print(remove_paths)
[shutil.move(f, os.path.join(data_path_lq, os.path.basename(f))) for f in remove_paths]
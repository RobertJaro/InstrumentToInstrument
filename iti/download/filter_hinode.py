import glob
import os
from matplotlib import pyplot as plt

files = glob.glob('/gss/r.jarolim/data/hinode/level1/*.fits')
invalid_names = [os.path.basename(f).replace('.jpg', '') for f in glob.glob('/gss/r.jarolim/data/hinode/invalid_img/*.jpg')]
to_remove = [f for f in files if os.path.basename(f) in invalid_names]
[os.remove(f) for f in to_remove]
import glob
import os

from astropy.io.fits import getheader
from tqdm import tqdm

path = '/gss/r.jarolim/data/soho/train'

eit_paths = [
    os.path.join(path, 'eit_171'),
    os.path.join(path, 'eit_195'),
    os.path.join(path, 'eit_284'),
    os.path.join(path, 'eit_304'),
    #os.path.join(path, 'mdi_mag')
]

files = [f for p in eit_paths for f in glob.glob(os.path.join(p, '*.fits'))]
for f in tqdm(files):
    header = getheader(f)
    valid = '  N_MISSING_BLOCKS =    0' in header['comment']
    if not valid:
        print(header['comment'])
        os.remove(f)


paths = [
    os.path.join(path, 'eit_171'),
    os.path.join(path, 'eit_195'),
    os.path.join(path, 'eit_284'),
    os.path.join(path, 'eit_304'),
    os.path.join(path, 'mdi_mag')
]
base_names = [[os.path.basename(f) for f in glob.glob(os.path.join(dir, '*.fits'))] for dir in paths]

intersection = set.intersection(*map(set,base_names))

all_files = glob.glob(os.path.join(path, '**/*.fits'))
to_remove = [f for f in all_files if os.path.basename(f) not in intersection]
[os.remove(f) for f in to_remove]
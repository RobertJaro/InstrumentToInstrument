import shutil
from urllib.request import urlretrieve

urlretrieve("https://kanzelhohe.uni-graz.at/iti/samples.zip", "samples.zip")
shutil.unpack_archive('samples.zip')
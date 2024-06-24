import setuptools
from setuptools import setup

setup(
    name='itipy',
    version='0.1',
    packages=setuptools.find_packages(),
    url='https://github.com/RobertJaro/InstrumentToInstrument',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Robert Jarolim',
    author_email='',
    description='Package for translation between image domains of different astrophysical instruments.',
    install_requires=['torch>=1.8', 'sunpy>=2.0', 'scikit-image', 'scikit-learn', 'tqdm',
                      'numpy', 'matplotlib', 'astropy', 'aiapy', 'drms', 'pytorch_fid']
)

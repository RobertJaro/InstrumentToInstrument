import setuptools
from setuptools import setup

setup(
    name='itipy',
    version='0.1',
    packages=setuptools.find_packages(),
    url='https://github.com/RobertJaro/InstrumentToInstrument',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Robert Jarolim',
    description='Package for translation between image domains of different astrophysical instruments.',
    install_requires=['torch>=1.8', 'sunpy>=2.0', 'scikit-image', 'scikit-learn', 'tqdm',
                      'numpy', 'matplotlib', 'astropy', 'aiapy','drms', 'jupyter', 'sunpy_soar',
                      'lightning', 'google', 'google-cloud-storage', 'wandb', 'pytorch_fid'],
    classifiers=[
    'Development Status :: 3 - Alpha', # either "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
    'License :: OSI Approved :: GPL-3.0 License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
  ],
)

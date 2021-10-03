import setuptools
from setuptools import setup, find_packages

setup(
    name='InstrumentToInstrument',
    version='0.1',
    packages=setuptools.find_packages(),
    url='https://github.com/RobertJaro/SolarInstrumentToInstrument',
    download_url='',
    license='GPL-3.0',
    author='Robert Jarolim',
    description='Package for translation between image domains of different astrophysical instruments.',
    install_requires=['torch>=1.8', 'sunpy>=2.0', 'scikit-image', 'scikit-learn', 'tqdm',
                      'numpy', 'matplotlib', 'astropy', 'aiapy','drms'],
    classifiers=[
    'Development Status :: 3 - Alpha', # either "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
    'License :: OSI Approved :: GPL-3.0 License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
  ],
)

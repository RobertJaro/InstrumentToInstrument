![](images/HinodeEnhanced_v2.jpg)
# Instrument to Instrument Translation for Solar Observations

# [Paper](#paper) --- [Usage](#usage) --- [Framework](#framework) 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertJaro/InstrumentToInstrument/blob/master/examples/ITI_translation.ipynb)

## Abstract
The constant improvement of astronomical instrumentation provides the foundation for scientific discoveries. In general, these improvements have only implications forward in time, while previous observations do not profit from this trend. In solar physics, the study of long-term evolution typically exceeds the lifetime of single instruments and data driven approaches are strongly limited in terms of coherent long-term data samples.

We demonstrate that the available data sets can directly profit from the most recent instrumental improvements and provide a so far unused resource to foster novel research and accelerate data driven studies.

Here we provide a general method that translates between image domains of different instruments (Instrument-to-Instrument translation; ITI), in order to inter-calibrate data sets, enhance physically relevant features which are otherwise beyond the diffraction limit of the telescope, mitigate atmospheric degradation effects and can estimate observables that are not covered by the instrument.

We demonstrate that our method can provide unified long-term data sets at the highest quality, by applying it to five different applications of ground- and space-based solar observations. We obtain 1) a homogeneous data series of 24 years of space-based observations of the solar corona, 2) solar full-disk observations with unprecedented spatial resolution, 3) real-time mitigation of atmospheric degradations in ground-based observations, 4) a uniform series of ground-based H-alpha observations starting from 1973, that unifies solar observations recorded on photographic film and CCD, 5) magnetic field estimates from the solar far-side based on multi-band EUV imagery. The direct comparison to simultaneous high-quality observations shows that our method produces images that are perceptually similar and match the reference image distribution.

## Usage

Start translating your own data online with Google Colab:
https://colab.research.google.com/github/RobertJaro/InstrumentToInstrument/blob/master/examples/ITI_translation.ipynb

For your local environment use pip:
``
pip install itipy
``

For GPU support follow the installation instructions of pytorch: https://pytorch.org/get-started/locally/


Google Colab offers free GPU resources, which can be used for a fast translation of data. 
Note that you have to upload and download your data to the Notebook. For HMI-to-Hinode continuum we provide a downloader that can be modified to translate custom observations. Data from SOHO and STEREO requires preprocessing routines, that are only available in SSW IDL. 
For larger amounts of data it is more efficient to translate the files on a local workstation (preferable with a GPU).

## Framework

Instrument-to-Instrument translation is designed as general framework that can be easily applied to similar tasks.
Many of the basic data loading, normalizing and scaling operations are already implemented by editors and can be used for the creation of new data sets, while also new custom editors can be added.


### Create custom data sets

ITI uses the pytorch Dataset class as basis for loading data.
The easiest way to create a new data sets is the use of existing data editors and the BaseDataset class.
The BaseDataset requires the path to the files or a list of file paths. 
The data processing pipeline can be customized by specifying editors, that will be sequentially applied to the data.
The first editor receives a file path. The output of each editor serve as input for the next editor.
In the example bellow we implement a custom data set for HMI magnetograms that:
1) loads a SunPy map from the file
2) centers the solar disk and scales the image to 2048
3) replaces all off-limb values with 0
4) Converts the SunPy map to a numpy array
5) replaces NaN values with 0
6) Scales the data to [-1, 1]
7) Reshapes the array to channel first notation

The editors are listed in ``iti.data.editor``. Custom editor (e.g., preprocessing) can be implement by using ``iti.data,editor.Editor`` as base class and implementing the call function.
Minor functionalities can be added by using ``iti.data,editor.LambdaEditor`` (e.g., ``LambdaEditor(lambda x: x * 2)``.

```python
from itipy.data.dataset import BaseDataset
from itipy.data.editor import LoadMapEditor, NormalizeRadiusEditor, RemoveOffLimbEditor, MapToDataEditor, NanEditor,

NormalizeEditor, ReshapeEditor
from astropy.visualization import ImageNormalize, LinearStretch


class HMIDataset(BaseDataset):

    def __init__(self, path, resolution=2048, ext='.fits', **kwargs):
        norm = ImageNormalize(vmin=-1000, vmax=1000, stretch=LinearStretch(), clip=True)
        editors = [
            # open FITS
            LoadMapEditor(),
            # normalize rad
            NormalizeRadiusEditor(resolution),
            # truncate off limb (optional)
            RemoveOffLimbEditor(),
            # get data from SunPy map
            MapToDataEditor(),
            # replace NaN with 0
            NanEditor(),
            # normalize data to [-1, 1]
            NormalizeEditor(norm),
            # change data to channel first format
            ReshapeEditor((1, resolution, resolution))]
        super().__init__(path, editors=editors, ext=ext, **kwargs)
```

### Create custom trainings

The trainer class implements a basic training function, that requires the workspace directory and the low- and high-quality dataset.
An optional validation data set can be specified to verify the model results (plots and parallel validation).

In the example bellow we translate SOHO observations to SDO quality. Both instruments have 5 channels (input_dim_a/b=5). We increase the resolution by a factor 2 (upsampling=1). 
We expect mostly instrumental characteristics that cause degradations and set the diversity factor to 0 (lambda_diversity=0).
For the training we specify the SDO and SOHO datasets, where we use a fixed resolution of 1024 pix for SOHO and consequently 2048 pix for SDO (1 times upsampling).
The training is performed with images patches, that we sample from the full-disk observations. According to our GPU memory we select a patch size of 128 pix for SOHO (256 pix for SDO).
We apply a temporal separation of our dataset, where we use the months 1-10 for training and 11-12 for validation.

Images are automatically saved during training, but note that they will only provide information about the qulaity of the 
translation when the InstanceNormalization weights are fixed (after 100 000 iterations).
The use of learned parameters of the InstanceNormalization is required for the training with image patches.

```python
from itipy.data.dataset import SDODataset, SOHODataset
from itipy.train.model import DiscriminatorMode
from itipy.trainer import Trainer

base_dir = ""
sdo_data_path = ""
soho_data_path = ""
# Init model
trainer = Trainer(input_dim_a=5, input_dim_b=5, upsampling=1, discriminator_mode=DiscriminatorMode.CHANNELS,
                  lambda_diversity=0)

# Init training datasets
sdo_train = SDODataset(sdo_data_path,
                       resolution=2048, patch_shape=(256, 256),
                       months=list(range(11)))
soho_train = SOHODataset(soho_data_path,
                         resolution=1024, patch_shape=(128, 128),
                         months=list(range(11)))
# Init validation/plotting datasets
sdo_valid = SDODataset(sdo_data_path,
                       resolution=2048, patch_shape=(256, 256),
                       months=[11, 12], limit=100)
soho_valid = SOHODataset(soho_data_path,
                         resolution=1024, patch_shape=(128, 128),
                         months=[11, 12], limit=100)

# Start training
trainer.startBasicTraining(base_dir,
                           ds_A=soho_train, ds_B=sdo_train,
                           ds_valid_A=soho_valid, ds_valid_B=sdo_valid)
```
# MLTRL Card -  - level 4

| Summary info        | Content, links       |
| -------------------------- | ------------- |
| Tech name (project ID)  | STEREO-to-SDO translation (ITI_1_0.1)   |
| Current Level           | 4 |
| Owner                   | Robert Jarolim                        |
| Project overview | https://spaceml.org/repo/project/6255ae968f8d1a000d74104f   |
| Code repo & docs        | https://github.com/RobertJaro/InstrumentToInstrument   |


**TL;DR** — STEREO-to-SDO translation is part of the ITI framework and provides  image enhancement, super-resolution and homogenization of STEREO observations to match the quality of SDO observations.

- **SDO** = Solar Dynamics Observatory; **AIA** = Atmospheric Imaging Assembly
- **STEREO** = Solar Terrestrial Relations Observatory; **EUVI** = Extreme Ultraviolet Imager

### Top-level requirements

1. The method shall translate STEREO/EUVI EUV filtergrams to SDO/AIA quality.
2. The method shall increase the spatial resolution (factor 4), enhance the appearance of solar features (sharpness), and adjust the calibration differences (photon counts).
4. The method shall provide a uniform series of EUV observations from multiple vantage points at the highest quality (three simultaneous observations).

### Model / algorithm info

Observations from SDO/AIA and STEREO/EUVI provide simultaneous observations of the Sun from three vantage points. 
The observations are different in their calibration and spatial resolution. To make use of the combined (highest quality) data series it is necessary to enhance STEREO/EUVI observations to SDO/AIA quality.
Instrument-To-Instrument translation (ITI) provides the ability to assimilate image domains without requiring a spatial overlap between the considered data sets.

Implementation notes:

- uses the ITI framework (unpaired image translation)
- simultaneous translation of all four wavelength channels
- training with 128x128 pixel patches and evaluation with full-disk images (random patches with bias for bright regions)
- uses the SDO-autocalibration for correction of device degradation
- STEREO device degradation is approximated by a first order fit of the quiet-Sun region for the 304 Å channel

### Intended use

- translate STEREO/EUVI filtergrams (171, 195, 284, 304 Å) to match characteristics of SDO/AIA filtergrams (171, 193, 211, 304 Å).
- apply automated methods developed for SDO to STEREO data.
- creation of inter-calibrated data series (e.g., use for solar cycle studies).
- image enhancement of STEREO filtergrams for better visualization.

**Extra notes**: observed wavelengths are no exact match and are consequently adjusted in the translation process.


### Testing status

- Comparison to a baseline calibration method shows that ITI provides a more consistent inter-calibration
- Verification of the FID shows a consistent quality improvement / data assimilation.
- No integration tests available.

### Data considerations

The data set comprises observations from the entire mission lifetime of each instrument. The evaluation and training was performed with real-world data (standard pre-processed).
For STEREO observations a daily sampling was used (A + B mixed). SDO observations were sampled with a 6 hour cadence (4 observations per day).
The standard test set split was used (test: November and December, train February-September).

Both the STEREO/EUVI and SDO/AIA calibration need to be consistent for the data sets (correction of device degradation).

Data handling for SDO and STEREO FITS files implemented.

Used files listed in `dataset/stereo.csv` and `dataset/sdo.csv`.

### Caveats, known edge cases, recommendations

- solar transient events (e.g., solar flares) are sparse in the data set and should be treated with caution.
- image edge regions (off-limb > 1.1 solar radii) can show artifacts (spurious emission)
- features beyond the resolution limit of STEREO/EUVI and with no connection to larger structures, shall not be considered for scientific evaluation.
- insufficient correction of device degradation for the individual data sets reduces the overall performance.

### MLTRL stage debrief

1. What was accomplished, and by who?

    - homogeneous series of solar EUV observations from three vantage points **See link to publication https://doi.org/10.21203/rs.3.rs-1021940/v1**.

3. What was learned?

    - correction of device degradation is an important pre-requirement. We plan to improve the correction for STEREO data at a later stage.
    - data set too large for online hosting
5. What tech debt was gained? Mitigated?

    - Colab notebook for online use of STEREO-To-SDO translation was added
    - integration tests are pending

---
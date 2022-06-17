# MLTRL Card -  - level 4

| Summary info        | Content, links       |
| -------------------------- | ------------- |
| Tech name (project ID)  | Instrument-to-Insturment translation (ITI_0_0.1)   |
| Current Level           | 4 |
| Owner                   | Robert Jarolim                        |
| Project overview        |    |
| Code repo & docs        | https://github.com/RobertJaro/InstrumentToInstrument   |


**TL;DR** — ITI is a framework for homogenizing data sets, stable enhancement of solar images and restoration of long-term image time series (iamge-enhancement; inter-calibration; unpaired image domain translation).


### Top-level requirements

1. The ITI method shall translate images from a 'low-quality' domain to a target 'high-quality' domain.
2. The ITI method shall use real observational data as reference for the image enhancement.
3. The translation shall not require any overlap between the considered instruments (spatial or temporal overlap)
4. The enhanced images shall be close to real 'high-quality' observations and shall have the same instrumental characteristics (e.g. calibration).
5. The method shall be flexible in terms of resolution differences (super-resolution), physical observables (wavelengths), type of quality degradation (atmospheric, instrumental-limitation) and field-of-view (full-disk, partial-Sun).


### Model / algorithm info

In solar physics, the study of long-term evolution typically exceeds the lifetime of single instruments. Data-driven approaches are limited in terms of homogeneous
historical data samples. Instrument-To-Instrument translation (ITI) can leverage recent instrumental improvements and provide a so far unused resource to foster novel research and accelerate data-driven solar research.

Implementation notes:

- unpaired image-to-image translation with Generative Adversarial Networks (GANs)
- includes noise-term for many-to-one mapping (noisy-to-enhanced)
- supports multiple observation channels (translated simultaneously)
- supports approximation of observables (e.g., different wavelengths)
- operates with full-resolution images
- general data pre-processing pipelines are provided (e.g., FITS reader, scale normalization)


**Extra notes**: The method is designed as a general framework for image domain translation tasks. We also provide model cards for the specific applications.

### Intended use


- translate between image domains of two (solar) instruments.
- the main application lies in the data calibration and assimilation, which allows studying unified data sets of multiple instruments and to apply automated algorithms to larger data sets without further adjustments or preprocessing. 
- the translation is performed for observations of the same type (e.g., LOS magnetogram) or in the same (or similar) wavelength band with similar temperature response (e.g., EUV).
- the instruments show a difference in quality (e.g., reduced by atmospheric conditions or by instrumental characteristics) and/or calibration.
- data sets need to provide similar observations in terms of features and regions (a different field-of-view is no limitation).
- paired images can be used but are not strictly required.

**Extra notes**: The estimation of observables (e.g., additional wavelengths) can be performed if the image translation is sufficiently constrained by the multi-channel context information and the learned high-quality image distribution. 


### Testing status

- Evaluation of five applications shows that the method is robust for various problem settings (image enhancement, inter-calibration, mitigation of atmospheric degradations, reconstruction of historical observations, approximation of observables).
- Verification of FIDs shows a consistent quality improvement / data assimilation.
- Evaluation with real-world data and comparison to baseline methods shows consistent high performance.
- Unpaired image translation approach shows comparable performance to paired image translations.
- Unit test for the framework components are pending.

**Extra notes**: Results of new applications should be verified independently before use.


### Data considerations

All trainings and evaluations have been performed with statistical accurate representation of real-world data.
Per default test data is selected from November and December of each year, while the training set corresponds to February-September.

The method is not considered to be operated with outliers in the data (e.g., severe image degradation, sparse events). 

Data handling for FITS files implemented. Extension to general data arrays (.npy files) and JPEGS is planned for the next version.

### Caveats, known edge cases, recommendations

- the model can draw information from the context at larger and intermediate scales, but the structures at the smallest scales can not be reconstructed (spatial information degraded beyond the point of reconstruction).
- enhanced images can be used for a better visualization of blurred features or to mitigate degradations, but they should not be used for studying small scale features.
- image degradations that are not present in the training set, are typically not accounted for.
- the high-quality data set (real observations) implies a upper limit of quality increase.

### MLTRL stage debrief

1. What was accomplished, and by who?

    - Investigated the general applicability of the method and the validity of the results **See link to publication https://doi.org/10.21203/rs.3.rs-1021940/v1**.

3. What was learned?

    - available data sets can directly profit from the most recent instrumental improvements via deep learning driven Instrument-To-Instrument translation
    - method can perform image enhancement, inter-calibration, mitigation of atmospheric degradations, reconstruction of historical observations, approximation of observables
5. What tech debt was gained? Mitigated?

    - examples and notebooks were added
    - unit tests and integration tests are pending

---
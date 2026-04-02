# Loop outputs

## Top-level outputs pre loop
Before the loop starts, a number of informational outputs are generated in the top-level loop folder.

These are:
- `config.yml`
    The configuration file used for the loop, containing all input parameters and settings.

- `gitl.log`
    Log file containing runtime information, warnings and diagnostic messages generated during execution.

## Per iteration outputs

The per-iteration outputs are saved to a folder named `iteration_XXXX` where `XXXX` is the iteration number, starting at 1.

The outputs within an iteration are:
- `efield_estimations.fits`
    Data cube of 2x3 frames containing the estimated electric field in the focal plane for each of the 3 wavelengths, real and imaginary. Sorting: R-W1, I-W1, R-W2, I-W2, R-W3, I-W3.
- `images.fits`
    All focal-plane images taken during the iteration. 21 images total: the first image is unprobed in wavelength 1, followed by 3 probe pairs (6 images total); then repeated for wavelength 2 and 3.
- `intensity_coherent.fits`
    Cube of 3 frames, per wavelength, containing the coherent intensity in the focal plane.
- `intensity_incoherent.fits`
    Cube of 3 frames, per wavelength, containing the incoherent intensity in the focal plane.
- `intensity_total.fits`
    Cube of 3 frames, per wavelength, containing the total intensity in the focal plane.
- `perfect_efields.fits`
    Data cube of 2x3 frames containing the perfect/model electric field in the focal plane for each of the 3 wavelengths, real and imaginary. Sorting: R-W1, I-W1, R-W2, I-W2, R-W3, I-W3.
- `svd_snorm.fits`
    Singular values squared, normalized by the maximum, ordered from largest to smallest.
- `svd_iri.fits`
    Power per singular-value mode, in the same order as `svd_snorm.fits`.
- `dm1_command.fits`
    Absolute DM1 voltage command for this iteration.
- `dm2_command.fits`
    Absolute DM2 voltage command for this iteration.

## Top-level outputs post loop
After the loop finishes, a number of outputs are generated containing initial analyses, results and metrics.

These are:
- `contrast_vs_iteration.pdf`
    Plot of measured and predicted broadband contrast vs iteration number.
- `ni_vs_iteration.pdf`
    Plot of normalized intensity (NI) metrics vs iteration number.
- `efield_variance.csv`
    Electric field variance per wavelength per iteration data as a csv table.
- `efield_variance.pdf`
    Plot of electric field variance per wavelength vs iteration number.
- `estimation_variance_per_pixel.fits`
    Estimation variance per pixel across all iterations, per wavelength, as a fits cube of 3 frames.
- `measured_contrast.csv`
    Measured broadband contrast per iteration as a csv table.
- `predicted_contrast.csv`
    Predicted contrast per iteration as a csv table.
- `debugging_history.csv`
    Per-wavelength debugging scalars appended each iteration. Only written if debugging data is available.
- `final_frames.fits`
    Final images taken after loop completion. 21 images total: the first image is unprobed in wavelength 1, followed by 3 probe pairs (6 images total); then repeated for wavelength 2 and 3.

## Example output directory structure

A typical HOWFSC loop run produces a directory with the following structure:

```
<run_directory>
├── config.yml
├── gitl.log
├── contrast_vs_iteration.pdf
├── ni_vs_iteration.pdf
├── efield_variance.csv
├── efield_variance.pdf
├── estimation_variance_per_pixel.fits
├── final_frames.fits
├── measured_contrast.csv
├── predicted_contrast.csv
├── debugging_history.csv
├── iteration_0001
│   ├── dm1_command.fits
│   ├── dm2_command.fits
│   ├── efield_estimations.fits
│   ├── images.fits
│   ├── intensity_coherent.fits
│   ├── intensity_incoherent.fits
│   ├── intensity_total.fits
│   ├── perfect_efields.fits
│   ├── svd_snorm.fits
│   └── svd_iri.fits
├── iteration_0002
│   └── ...
├── ...
└── iteration_XXXX
    └── ...
```

Where:

- `<run_directory>` is automatically created for each loop run (typically including a timestamp and model name).

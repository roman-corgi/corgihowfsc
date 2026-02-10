# Loop outputs

## Top-level outputs pre loop
Before the loop starts, a number of informational outputs are generataed in the top-level loop folder.

These are:
- N/A for now

## Per iteration outputs

The per-iteration outputs are saved to a folder named `iteration_XXXX` where `XXXX` is the iteration number, starting at 1.

The outputs within an iteration are:
- `efield_estimations.fits`
    Data cube of 2x3 frames containing the estimated electric field in the focal plane for each of the 3 wavelengths, real and imaginary. Sorting: R-W1, I-W1, R-W2, I-W2, R-W3, I-W3.
- `images.fits`
    All focal-plane images taken during the iteration. 21 images total: the first images is unprobed in wavelength 1, followed by 3 probe pairs (6 images total); then repeated for wavelength 2 and 3.
- `intensity_coherent.fits`
    Cube of 3 frames, per wavelength, containing the coherent intensity in the focal plane.
- `intensity_incoherent.fits`
    Cube of 3 frames, per wavelength, containing the incoherent intensity in the focal plane.
- `intensity_total.fits`
    Cube of 3 frames, per wavelength, containing the total intensity in the focal plane.
- `dm_data`
    Directory containing the DM commands per iteration, saved as `dm1_command_history.fits` and `dm2_command_history.fits`, which get updated during each iteration. If N total iterations were run, the fits files contaion N+1 frames, the first one being from before the loop starts.

## Top-level outputs post loop
After the loop finishes, a number of outputs are generated, containing initial analyses, results and metrics.

These are:
- `contrast_vs_iteration.pdf`
    Plot of composed broadband contrast in three wavelengths vs iteration number.
- `measured_contrast.csv`
    Composed broadband contrast per iteration data as csv table.
- `final_frames.fits`
    Final images taken after loop completion. 21 images total: the first images is unprobed in wavelength 1, followed by 3 probe pairs (6 images total); then repeated for wavelength 2 and 3.
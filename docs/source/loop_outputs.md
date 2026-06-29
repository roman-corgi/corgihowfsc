# Loop outputs

## Top-level outputs pre loop
Before the loop starts, a number of informational outputs are generated in the top-level loop folder.

These are:
- `config.yml`
    The configuration file used for the loop, containing all input parameters and settings.

- `gitl.log`
    Log file containing runtime information, warnings and diagnostic messages generated during execution.

- `image_worker_debug.csv`
    Optional debug CSV written when debug mode is enabled.

- `gitl_rank<rank>.log`
    Optional MPI worker log files written when both debug mode and MPI are enabled.

## Per iteration outputs

The per-iteration outputs are saved to a folder named `iteration_XXXX` where `XXXX` is the iteration number, starting at 1.

The outputs within an iteration are:
- `efield_estimations.fits`
    Data cube containing the estimated electric field in the focal plane for each wavelength channel, stored as real and imaginary. The total number of planes is `2 * nlam`.
    Example: in NFOV band 1 with `nlam = 3`, the file contains 6 planes ordered as `R-W1, I-W1, R-W2, I-W2, R-W3, I-W3`.
- `images.fits`
    All focal-plane images taken during the iteration. The total number of images is `nlam * ndm`, where `ndm = 1 + 2 * nprobepair` for one unprobed image plus positive/negative probe pairs at each wavelength.
    Example: in NFOV band 1 with `nlam = 3` and `nprobepair = 3`, `ndm = 7` and the file contains `3 * 7 = 21` images.
- `intensity_coherent.fits`
    Cube of `nlam` frames, one per wavelength, containing the coherent intensity in the focal plane.
- `intensity_incoherent.fits`
    Cube of `nlam` frames, one per wavelength, containing the incoherent intensity in the focal plane.
- `intensity_total.fits`
    Cube of `nlam` frames, one per wavelength, containing the total intensity in the focal plane.
- `perfect_efields.fits`
    Data cube containing the perfect/model electric field in the focal plane for each wavelength channel, stored as real and imaginary. When model e-fields are available for all wavelengths, the total number of planes is `2 * nlam`.
    Example: in NFOV band 1 with `nlam = 3`, the file contains 6 planes ordered as `R-W1, I-W1, R-W2, I-W2, R-W3, I-W3`.
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
    Estimation variance per pixel across all iterations, per wavelength, stored as a fits cube with one frame per wavelength.
- `measured_contrast.csv`
    Measured broadband contrast per iteration as a csv table.
- `predicted_contrast.csv`
    Predicted contrast per iteration as a csv table.
- `debugging_history.csv`
    Per-wavelength debugging scalars appended each iteration. Only written if debugging data is available.
- `final_frames.fits`
    Final images taken after loop completion. The total number of images is `nlam * ndm`, where `ndm = 1 + 2 * nprobepair`.
    Example: in NFOV band 1 with `nlam = 3` and `nprobepair = 3`, this is 21 images.

## Debugging History CSV Details

The `debugging_history.csv` file is written at the end of each HOWFSC iteration
and records per-iteration, per-wavelength diagnostic quantities from the GITL loop.
One row is written per wavelength channel per iteration, so a run with `N` iterations
and `L` wavelength channels produces `N × L` rows.

### File Format

The file has no header on row 0 (which contains metadata); the column names appear on
row 1. It can be loaded with pandas as follows:

```python
import pandas as pd
df = pd.read_csv("debugging_history.csv", skiprows=1)
```

Alternatively, use the provided helper:

```python
from corgihowfsc.analysis import load_debugging_csv
history = load_debugging_csv("debugging_history.csv")
```

The helper returns a dictionary of the form `history[field][lam_index]`, where each
value is a 1-D array of length `N` (one entry per iteration). Iteration indexing in
the raw CSV is 1-based.

### Columns

| Column | Units | Description |
|--------|-------|-------------|
| `iteration` | — | 1-based iteration counter. |
| `lam_index` | — | 0-based wavelength channel index. |
| `beta` | — | Log₁₀ of the EFC regularization parameter selected for this iteration. Less negative values (e.g. `-3.5`) apply weaker regularization; more negative values (e.g. `-6.5`) apply stronger regularization. The value is chosen adaptively by the [Control Strategy Configuration](cstrat_docs.md). |
| `peakflux` | counts / s | Unocculted stellar peak flux used to normalize raw camera images to normalized intensity (NI) units. Derived from the EETC for the current stellar type and magnitude. |
| `next_c` | contrast | Mean dark hole contrast measured at the end of this iteration (i.e. the starting contrast for the next iteration). |
| `this_iter_dur` | s | Wall-clock duration of this iteration, including probing, estimation, and control computation. This is the same for all wavelength channels within a given iteration. Cumulative sum OF ONE SUBBAND gives total elapsed time. |
| `this_iter_dur_gitl` | s | Wall-clock duration of the GITL-specific overhead within this iteration. |
| `cam_nom_gain` | — | EM gain setting used for the unocculted (nominal) observation. |
| `cam_nom_exptime` | s | Exposure time used for the unocculted (nominal) observation. |
| `cam_nom_nframes` | — | Number of frames co-added for the unocculted (nominal) observation. |
| `cam_probe_gain` | — | EM gain setting used for the probed observations. |
| `cam_probe_exptime` | s | Exposure time used for each probed observation. |
| `cam_probe_nframes` | — | Number of frames co-added for each probed observation. |
| `pred_mean_contrast` | contrast | EFC-predicted mean dark hole contrast after applying the DM update, computed from the estimated electric field. |
| `pred_bright_contrast` | contrast | EFC-predicted brightest-pixel dark hole contrast after applying the DM update. |
| `pred_mean_contrast_probing` | contrast | Predicted mean dark hole contrast evaluated at the probing DM state (before the EFC correction is applied). |
| `pred_bright_contrast_probing` | contrast | Predicted brightest-pixel dark hole contrast evaluated at the probing DM state. |

### Usage Notes

#### Cumulative elapsed time

Because `this_iter_dur` is identical across wavelength channels for a given
iteration, extract it from a single channel before taking the cumulative sum:

```python
history = load_debugging_csv("debugging_history.csv")
iter_dur = np.array(history['this_iter_dur'][0])   # lam_index=0
cumtime_hr = np.cumsum(iter_dur) / 3600.0
```

#### NI conversion for raw images

The `RAW_IMAGES` layer in `images.fits` is in detector counts and must be
converted to NI before comparison with contrast metrics in this file:

```python
NI = raw_image / cam_nom_exptime / peakflux
```

The `intensity_coherent` and `intensity_incoherent` FITS layers are already
in NI units and require no further conversion.

#### Relationship to `measured_contrast.csv`

`measured_contrast.csv` contains one row per iteration (averaged over wavelength
channels) and is the primary source for contrast vs. iteration plots.
`debugging_history.csv` provides the per-wavelength, per-iteration detail needed
for timing analysis, camera parameter inspection, and prediction vs. measurement
comparisons.


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

# Loop outputs

## Top-level outputs pre loop
Before the loop starts, a number of informational outputs are generataed in the top-level loop folder.

These are:
- N/A for now

## Per iteration outputs

The per-iteration outputs are saved to a folder named `iteration_XXXX` where `XXXX` is the iteration number, starting at 1.

The outputs within an iteration are:
- `efield_estimations.fits`
- `images.fits`
- `intensity_coherent.fits`
- `intensity_incoherent.fits`
- `intensity_total.fits`

## Top-level outputs post loop
After the loop finishes, a number of outputs are generated, containing initial analyses, results and metrics.

These are:
- `contrast_vs_iteration.pdf`
- `measured_contrast.csv`
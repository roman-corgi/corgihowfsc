# Jacobian computation

The original `cgi-howfsc` loop code comes with several options on when and how to compute the Jacobian matrix and related data.
These can be called in the same way on the `corgihowfsc` repo, where the relevant loop code is encapsulated in the function 
`nulling_gitl()`, located in `corgihowfsc/gitl/nulling_gitl.py`. The original function on `cgi-howfsc` that does this can be found
[here](https://github.com/roman-corgi/cgi-howfsc/blob/0a3a3f1439eb5db4dffd4ae69187f5c4ca1ed12f/howfsc/scripts/nulltest_gitl.py#L43).

The files relating to a Jacobian input are:
- The Jacobian matrix
- The JTWJ map, which is the weighting map `W` (`J^T * W * J`) as it is used in the EFC algorithm; more info [here](https://github.com/roman-corgi/cgi-howfsc/blob/0a3a3f1439eb5db4dffd4ae69187f5c4ca1ed12f/howfsc/control/calcjtwj.py#L16).
- The n2clist, which is a list of conversion factors from normalized intensity to contrast. For more info see [here](https://github.com/roman-corgi/cgi-howfsc/blob/0a3a3f1439eb5db4dffd4ae69187f5c4ca1ed12f/howfsc/control/nextiter.py#L55).

## Jacobian strategies

The different Jacobian strategies represent different choice for when which of the above files are precomputed for the loop and loaded from disk, or computed at runtime.

:::{note}
There are four Jacobian strategies for the HOWFSC loop:

| Jacobian and JTWJ                               | n2clist                                         | keyword               |
|-------------------------------------------------|-------------------------------------------------|-----------------------|
| Load from disk, keep fixed throughout loop      | Load from disk, keep fixed throughout loop      | `load_all`            |
| Compute before loop, keep fixed throughout loop | Load from disk, keep fixed throughout loop      | `precomp_jacs_once`   |
| Compute before each iteration                   | Load from disk, keep fixed throughout loop      | `precomp_jacs_always` |
| Compute before loop, keep fixed throughout loop | Compute before loop, keep fixed throughout loop | `precomp_all_once`    |
:::

The above are the three options fed into the `precomp` variable of the top-level HOWFSC function called `nulling_gitl()` called in the launcher script.

:::{important}
CGI is anticipated to calculate Jacobians at each iteration, which is why this is the default option in `corgihowfsc`.
:::

**Original docstring from `git_howfsc` repo:**  
See original docstrings in the `roman-corgi/cgi-howfsc` repo [here](https://github.com/roman-corgi/cgi-howfsc/blob/0a3a3f1439eb5db4dffd4ae69187f5c4ca1ed12f/howfsc/scripts/nulltest_gitl.py#L91).

```yaml
precomp : str, optional
    One of 'load_all', 'precomp_jacs_once', 'precomp_jacs_always',
    or 'precomp_all_once'.  This determines how the Jacobians and related
    data are handled.  Defaults to 'load_all'.
    'load_all' means that the Jacobians, JTWJ map, and n2clist are all
    loaded from files in jacpath and leaves them fixed throughout a loop.
    'precomp_jacs_once' means that the Jacobians and JTWJ map are computed
    once at the start of the sequence, and the n2clist is loaded from files
    in jacpath.
    'precomp_jacs_always' means that the Jacobians and JTWJ map are computed
    at the start of the sequence and then recomputed at the start of every
    iteration except the last one; the n2clist is loaded from files in jacpath.
    'precomp_all_once' means that the Jacobians, JTWJ map, and n2clist are
    all computed once at the start of the sequence.
```

## Precomputing a Jacobian

To calculate a Jacobian outside of the loop, use `corgihowfsc/scripts/make_jacobian.py`.
:::{note}
The Jacobian is computed using the compact HOWFSC model configured for the
selected `corgihowfsc` mode and dark-hole setting.
:::

### Basic usage

```bash
python corgihowfsc/scripts/make_jacobian.py \
  --mode nfov_band1 \
  --dark_hole 360deg \
  --jacmethod fast \
  --num_process 0 \
  --num_threads 1
```

By default, output is written under:

```text
~/corgiloop_data/jacobians/
```

`--num_process 0` means "use half of the available CPU cores."

### Overriding the DM operating point

By default, `make_jacobian.py` uses the mode-specific DM start maps returned by
`load_files(...)`. You can override that operating point with `--dm1_start` and
`--dm2_start` if you want to linearize the Jacobian around a different DM state.

This is useful, for example, when regenerating a Jacobian after a GITL run for
a specific iteration using the DM commands from that iteration.

```bash
python corgihowfsc/scripts/make_jacobian.py \
  --mode nfov_band1 \
  --dark_hole 360deg \
  --jacmethod fast \
  --dm1_start /path/to/dm1.fits \
  --dm2_start /path/to/dm2.fits
```

:::{important}
`--dm1_start` and `--dm2_start` must be provided together, since they define
the DM operating point used for Jacobian linearization.
:::

Each override may be either:
- an absolute path, or
- a filename relative to the selected model directory

### Output files

The output file size is model-dependent and is determined primarily by the
number of dark-hole pixels included in the selected model.
Compared with the current `corgihowfsc` `nfov_band1` setup, the legacy
`cgi-howfsc` `narrowfov` model uses dark-hole masks with many more pixels, so
its saved Jacobian files are correspondingly larger.

The generated FITS file contains the Jacobian matrix only. If you want the
associated JTWJ map or n2clist, those are still handled separately by the loop
precomputation logic.

## Implementation details

A bunch of implementation caveats for the `corgihowfsc` repo (TBD).
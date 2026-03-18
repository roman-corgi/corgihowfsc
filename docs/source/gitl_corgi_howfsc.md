# corgi-howfsc loop

This example shows how to run the baseline GITL nulling test with the `corgi-howfsc` code developped by the CPP.
All examples are set up to run on the NFOV HLC mode.

:::{important}
The corgihowfsc loop can be run with **either the corgisim model or the compact model**.
:::

The `corgi-howfsc` inherits from `cgi-howfsc` (Roman CPP fork), which contains a "compact" model of a coronagraph 
instrument that is used to calculate a Jacobian to use on the Coronagraph Instrument. 

`corgi-howfsc` also inherits from `corgisim` which is used to generate flight-like images. 
Even in this case the Jacobian is always calculated using the compact model from `cgi-howfsc`, 
which is the one that will be used in flight.

:::{important}
All examples are set up to run on the NFOV HLC mode.  
All code examples are runnable in the corgiloop conda env of corgihowfsc.
:::

If you wish to precompute a Jacobian instead of calculating it at runtime, [check here on how to calculate a Jacobian in corgihowfsc](jacobian_computation.md#precomputing-a-jacobian).

:::{important}
The launcher script for running a corgihowfsc loop is always `scripts/run_corgisim_nulling_gitl.py`, independent of optical model choice.
:::

## Run a nulling test with the **compact** model

There is one single parameter that differentiates between running your loop with the compact model or the full `corgisim` model.
To run with the compact model, set the `backend_type` variable to `cgi-howfsc` (the default is `cgi-howfsc`):
```python
backend_type = 'cgi-howfsc'  # 'corgihowfsc' for the corgisim model, otherwise for the compact model use: 'cgi-howfsc'
```
In this situation both the Jacobian and the simulated images are generated from the compact model. 
This is the fastest mode to run.

## Run a nulling test with the **corgisim** model

There is one single parameter that differentiates between running your loop with the compact model or the full `corgisim` model.
To run with the corgisim model, set the `backend_type` variable to `corgihowfsc` (the default is `cgi-howfsc`):
```python
backend_type = 'corgihowfsc'  # 'corgihowfsc' for the corgisim model, otherwise for the compact model use: 'cgi-howfsc'
```

In this situation, the Jacobian is still calculated from [the compact model](jacobian_computation.md#precomputing-a-jacobian) but the images are generated
using the realistic `corgisim` model.  Note that this mode is very slow to run 
(typically several minutes per iteration on a laptop). 

When running with `corgisim` the user can also choose the normalization strategy via which we go from photoelectrons to contrast units. The default normalization strategy is `eetc` which uses the official engineering exposure time calculator to determine the peakflux. Other options include:
```python
normalization_type = 'eetc'
# place off-axis source at 7L/D to get peakflux: 'corgisim-off-axis'
# Take FPM out of beam to get peakflux: 'corgisim-on-axis'
```

## Common parameters between optical models

You can run the `scripts/run_corgisim_nulling_gitl.py` script as follows by passing the path to your Jacobian (optional),
in which case you need to set the `precomp` variable to `load_all`:
```python

defjacpath = '/Users/user/data_from_repos/corgiloop/jacobians'
precomp = 'load_all'
```
If the Jacobian is not pre-computed, users can leave the default as:
```python
defjacpath = os.path.join(os.path.dirname(howfscpath), 'temp')
precomp = 'precomp_jacs_always'
```
User can also update the output path, otherwise all loop outputs will be go throuhg a dedicated folder in your home directory:
```python
base_path = Path.home()  # this is the proposed default but can be changed
```
The initial DM settings are by default set to the NFOV band 1 seed commands as:
```python
dmstartmap_filenames = ['iter_080_dm1.fits', 'iter_080_dm2.fits']
```
For WFOV band 4 use:
```python
dmstartmap_filenames = ['iter_061_dm1.fits', 'iter_061_dm2.fits']
```

For parallel computing the parameters are: 
```python
num_proper_process = 5 
num_jac_process = 12 
num_imager_worker = None
```
If running on a powerful desktop or cluster, the user can set `num_imager_worker` to an integer >1 such that multiple probe images are simulated in parallel.


From here the script can be run as-is! The result will be some iteration-specific information printed to stdout, and a
suite of output files being saved to the output folder [see loop outputs](loop_outputs.md).

## Inside of main():
From here, all of the configuration files are loaded. 
1. We start with `get_args` which builds the `args` class which contains a number of basic parameters
determined by the FPM, bandpass, probe shape, dark hole shape, initial DM shapes, and Jacobian calculation properties.
2. Next we load the files or get the full filepaths for all of the necessary yaml and fits files we need using `load_files`
3. Using the coronagraph configuration filepath, we generate the corongraph configuration object `cfg`
4. Using the hconf filepath, generate the `hconf` object 
    * This is where the `sequence` used by `EETC` is used
    * This is where the stellar type and spectrum is set. If the user would like to change those, they can either be changed in the hconf file or:

```python
hconf = loadyaml(hconffile, custom_exception=TypeError)
hconf['star']['stellar_vmag'] = 5
hconf['star']['stellar_type'] = 'G05'
```
5. Next we load the control strategy file to generate the `cstrat` object
   * This is where the probe amplitude schedule and EFC regularization schedules are defined
6. Define the `esitmator`
7. Define the `probes` class
8. Define `imager` class
   
**Note** any `proper` keyword can be passed through `corgi_overrides` as long as the key used in `corgi_overrides` is the same as the key for the relevant `proper_keyword`.

If user would like to compare the output with the compact model, the recommended method is to 
use [corgi-howfs with the compact model](#run-a-nulling-test-on-compact-model).

However, it is still possible to use the compact model directly in `cgi_howfs`. 
This is **not** the recommended method and this should be reserved for particular situations.  
Please refer to [cgi_howfs GITL (Compact)](gitl_cgi_howfsc.md).

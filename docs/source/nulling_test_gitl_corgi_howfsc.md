# GITL with corgi-howfsc loop

This example shows how to run the baseline GITL nulling test with the `corgi-howfsc` code developped by the CPP.
All examples are set up to run on the NFOV HLC mode.

.. important::
    The corgihowfsc loop can be run with **either the corgisim model or the compact model**.

The `corgi-howfsc` inherits from `cgi-howfsc` (Roman CPP fork), which contains a "compact" model of a coronagraph 
instrument that is used to calculate a Jacobian to use on the Coronagraph Instrument. 

`corgi-howfsc` also inherits from `corgisim` which is used to generate flight-like images. 
Even in this case the Jacobian is always calculated using the compact model from `cgi-howfsc`, 
which is the one that will be used in flight 

## Calculate a Jacobian
To calculate a Jacobian, you can call the function from `cgi-howfsc` that does this:

```python
from howfsc.scripts.jactest_mp import calculate_jacobian_multiprocessed

output = '/Users/user/directory/first_jacobian.fits'
calculate_jacobian_multiprocessed(output=output, proc=0)
```

The resulting file has a size of 2.2 GB. The function docstring contains more information about the input parameters.

## Run a nulling test on compact model

In this situation both the Jacobian and the simulated images are generated from the compact model. 
This is the fastest mode to run. 

Optional:
You will need to rename your Jacobian for the respective coronagraph mode you want to run a loop on. For the narrow FOV mode
with the HLC, rename the Jacobian to `narrowfov_jac_full.fits`.

Then you can run the `scripts/run_nulling_gitl.py` script as follows by passing the path to your Jacobian, and output paths:

```python

defjacpath = '/Users/user/data_from_repos/corgiloop/jacobians'
precomp = 'load_all'
```
If the Jacobian is not pre-computed, users can leave the default as:
```python
defjacpath = os.path.join(os.path.dirname(howfscpath), 'temp')
precomp = 'precomp_jacs_always'
```
User can also update the output file path and name:
```python
current_datetime = datetime.now()
folder_name = 'gitl_simulation_' + current_datetime.strftime("%Y-%m-%d_%H%M%S")
fits_name = 'final_frames.fits'
fileout_path = os.path.join(os.path.dirname(os.path.dirname(corgihowfsc.__file__)), 'data', folder_name, fits_name)
```
The initial DM settings are by default set as:
```python
dmstartmap_filenames = ['iter_080_dm1.fits', 'iter_080_dm2.fits']
```
From here the script can be run as-is! The result will be some iteration-specific information printed to stdout, and a `fileout.fits` file containing the
results of the final iteration of the loop.

## Run a nulling test on corgisim model

In this situation, the Jacobian is still calculated form [the compact model](#calculate_jac) but the images are generated
using the realistic `corgisim` model.  Note that this mode is very slow to run 
(typically several minutes per iteration on a laptop). 

Optional:
You will need to rename your Jacobian for the respective coronagraph mode you want to run a loop on. For the narrow FOV mode
with the HLC, rename the Jacobian to `narrowfov_jac_full.fits`.

Then you can run the `scripts/run_nulling_gitl.py` script as follows by passing the path to your Jacobian, and output paths:

```python

defjacpath = '/Users/user/data_from_repos/corgiloop/jacobians'
precomp = 'load_all'
```
If the Jacobian is not pre-computed, users can leave the default as:
```python
defjacpath = os.path.join(os.path.dirname(howfscpath), 'temp')
precomp = 'precomp_jacs_always'
```
User can also update the output file path and name:
```python
current_datetime = datetime.now()
folder_name = 'gitl_simulation_' + current_datetime.strftime("%Y-%m-%d_%H%M%S")
fits_name = 'final_frames.fits'
fileout_path = os.path.join(os.path.dirname(os.path.dirname(corgihowfsc.__file__)), 'data', folder_name, fits_name)
```
The initial DM settings are by default set as:
```python
dmstartmap_filenames = ['iter_080_dm1.fits', 'iter_080_dm2.fits']
```

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

If user would like to compare the output with the compact model, the recommended method is to 
use [corgi-howfs with the compact model](#compact_model)

However, it is still possible to use the compact model directly in `cgi_howfs`. 
This is **not** the recommended method and this should be reserved for particular situations.  
please refer to [cgi_howfs GITL(Compact)](nulling_test_gitl_cgi_howfsc.md)

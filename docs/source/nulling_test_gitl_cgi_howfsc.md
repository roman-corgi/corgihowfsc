# GITL directly with cgi-howfsc loop (dev only)

:::{warning}
We do *not* want to run cgi-howfsc code directly. This page exists purely for information purposes.
:::

This example shows how to run the baseline GITL nulling test with the `cgi-howfsc` code as published by NASA.
This mode is superceded by the implementation of the `cgi-howfsc` compact model in `corgi_howfsc`.  
Instructions for this mode can be found [here](nulling_test_gitl_corgi_howfsc.md)
All examples are set up to run on the NFOV HLC mode.

:::{important}
The cgi-howfsc loop can only be run on the **compact model**.
:::

The original code was published under [https://github.com/nasa-jpl/cgi-howfsc](https://github.com/nasa-jpl/cgi-howfsc).
For CPP work, it was decided to fork this repository into [https://github.com/roman-corgi/cgi-howfsc](https://github.com/roman-corgi/cgi-howfsc)
to keep a track of the separate development done by the CPP team. The forked repository is the one used in `corgihowfsc`.

The `cgi-howfsc` repository contains a "compact" model of a coronagraph instrument, which is used to calculate a Jacobian
to use on the Coronagraph Instrument.

## Calculate a Jacobian

To calculate a Jacobian, you can call the function from cgi-howfsc that does this:

```python
from howfsc.scripts.jactest_mp import calculate_jacobian_multiprocessed

output = '/Users/user/directory/first_jacobian.fits'
calculate_jacobian_multiprocessed(output=output, proc=0)
```

The resulting file has a size of 2.2 GB. The function docstring contains more information about the input parameters.

## Run a nulling test on compact model

You will need to rename your Jacobian for the respective coronagraph mode you want to run a loop on. For the narrof FOV mode
with the HLC, rename the Jacobian to `narrowfov_jac_full.fits`.

Then you can run the GITL nulling test as follows by passing the path to your Jacobian, and output paths:

```python
from howfsc.scripts.nulltest_gitl import nulling_test_gitl

logfile = '/Users/user/data_from_repos/corgiloop/loop1/logging.log'
fileout = '/Users/user/data_from_repos/corgiloop/loop1/fileout.fits'
jacpath = '/Users/user/data_from_repos/corgiloop/jacobians'

nulling_test_gitl(logfile=logfile, fileout=fileout, jacpath=jacpath)
```

The result will be some iteration-specific information printed to stdout, and a `fileout.fits` file containing the
results of the final iteration of the loop.

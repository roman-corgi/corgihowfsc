# cgi-howfsc loop directly (dev only)

:::{warning}
We do *not* want to run cgi-howfsc code directly.
This page exists purely for information purposes.
:::

This doc page refers to using the direct cgi-howfsc launcher in `scripts/run_corgisim_nulling_gitl.py`.

This example shows how to run the baseline GITL nulling test with the `cgi-howfsc` code as published by NASA.
This mode is superceded by the implementation of the `cgi-howfsc` compact model in the `corgi_howfsc` repository.  
Instructions for running loops with corgihowfsc (on either optical model, compact or corgisim) can be found [here](gitl_corgi_howfsc.md).

:::{important}
The cgi-howfsc loop can only be run on the **compact model**.
:::

The original `cgi-howfsc` code was published under [https://github.com/nasa-jpl/cgi-howfsc](https://github.com/nasa-jpl/cgi-howfsc).
For CPP work, it was decided to fork this repository into [https://github.com/roman-corgi/cgi-howfsc](https://github.com/roman-corgi/cgi-howfsc)
to keep a track of the separate development done by the CPP team. The forked repository is the one imported in `corgihowfsc`.

The `cgi-howfsc` repository contains a "compact" model of a coronagraph instrument, which is used to calculate a Jacobian
to use on the Coronagraph Instrument.

:::{important}
All examples are set up to run on the NFOV HLC mode.  
All code examples are runnable in the corgiloop conda env of corgihowfsc.
:::

## Run a nulling test on compact model

If you wish to precompute a Jacobian instead of calculating it at runtime, [check here on how to calculate a Jacobian in corgihowfsc](jacobian_computation.md#precomputing-a-jacobian).

You will need to rename your Jacobian for the respective coronagraph mode you want to run a loop on. For the narrow FOV mode
with the HLC, rename the Jacobian to `narrowfov_jac_full.fits`.

Then you can run the GITL nulling test as follows by passing the path to your Jacobian, and output paths:

```python
from howfsc.scripts.nulltest_gitl import nulling_test_gitl

logfile = '/Users/user/data_from_repos/corgiloop/loop1/logging.log'
fileout = '/Users/user/data_from_repos/corgiloop/loop1/fileout.fits'
jacpath = '/Users/user/data_from_repos/corgiloop/jacobians'

nulling_test_gitl(logfile=logfile, fileout=fileout, jacpath=jacpath)
```

The result will be some iteration-specific information printed to stdout, and a suite of output files being saved to the output folder [see loop outputs](loop_outputs.md).

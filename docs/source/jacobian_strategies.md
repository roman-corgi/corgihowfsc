# Jacobian strategies

The original `cgi-howfsc` loop code comes with several options on when and how to compute the Jacobian matrix and related data.
These can be called in the same way on the `corgihowfsc` repo, where the relevant loop code is encapsulated in the function 
`nulling_gitl()`, located in `corgihowfsc/gitl/nulling_gitl.py`. The original function on `cgi-howfsc` that does this can be found
[here](https://github.com/roman-corgi/cgi-howfsc/blob/0a3a3f1439eb5db4dffd4ae69187f5c4ca1ed12f/howfsc/scripts/nulltest_gitl.py#L43).

The files relating to a Jacobian input are:
- The Jacobian matrix
- The JTWJ map, which is the weighting map `W` (`J^T * W * J`) as it is used in the EFC algorithm; more info [here](https://github.com/roman-corgi/cgi-howfsc/blob/0a3a3f1439eb5db4dffd4ae69187f5c4ca1ed12f/howfsc/control/calcjtwj.py#L16).
- The n2clist, which is a list of conversion factors from normalized intensity to contrast. For more info see [here](https://github.com/roman-corgi/cgi-howfsc/blob/0a3a3f1439eb5db4dffd4ae69187f5c4ca1ed12f/howfsc/control/nextiter.py#L55).

The different Jacobian strategies represent different choice for when which of the above files are precomputed for the loop and loaded from disk, or computed at runtime.

:::{note}
There are three Jacobian strategies for the HOWFSC loop:
- Load Jacobian and related files from disk and keep them fixed throughout the loop (`load_all`).
- Precompute the Jacobian and related files once at the start of the loop, and then keep them fixed throughout the loop (`precomp_jacs_once`).
- Compute the Jacobian and related files at the start of each iteration (`precomp_jacs_always`).
:::

**Original docstring from `git_howfsc` repo:**  
See original docstrings in the `roman-corgi/cgi-howfsc` repo [here](https://github.com/roman-corgi/cgi-howfsc/blob/0a3a3f1439eb5db4dffd4ae69187f5c4ca1ed12f/howfsc/scripts/nulltest_gitl.py#L91).

:::{warning}
The below docstring is currently ambiguous and we are confirming with AJ that we understood correctly.
:::

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


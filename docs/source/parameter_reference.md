# Parameter Reference

This page documents every field in the loop parameter file `corgihowfsc/scripts/default_param.yml`.

Pass the file to the launcher script via:

```bash
python scripts/run_corgisim_nulling_gitl.py --param_file /path/to/params.yml
```

If `--param_file` is not provided, the script falls back to `default_param.yml` in the same directory.

---

## Full default parameter file

```yaml
active_model: "cgi-howfsc" # choose from "cgi-howfsc" or "corgihowfsc". This will determine which imager model is used and which settings are applied to the imager initialization.

runtime:
  num_proper_process: 5 # default to 5
  num_jac_process: 6 # default to 2
  num_imager_worker: null # default to null for no parallelization, otherwise set to an integer number of workers
  use_mpi: false # or null if mpi is false. Set to true to use MPI for parallelization.

sim_settings:
  loop_framework: "corgi-howfsc" # do not modify
  precomp: "precomp_jacs_always" # options: 'precomp_jacs_always', 'precomp_all_once', 'load_all' (only if defjacpath is not None)
  output_every_iter: true # whether to save the output frames at every iteration (true) or after all iterations are done (false)
  niter: 3 # number of iterations to run
  mode: "nfov_band1" # options: see models
  dark_hole: "360deg"  # options: see models
  probe_shape: "default"

paths:
  base_path: "~" # default to home directory, but user can change to a different path
  base_corgiloop_path: "corgiloop_data" # this is the proposed default but can be changed
  defjacpath: "temp" # this is the proposed default but can be changed, should be somewhere outside the repo if used
  final_filename: "final_frames.fits" # this is the proposed default but can be changed
  folder_tag: null # this is an optional string that will be appended to the folder name where results are saved

path_overrides:
  # Set any of these to an absolute path to override the mode-derived default.
  # Leave as null to use the mode-derived path.
  # Can also override defjacpath and dmstartmap_filenames by providing full paths
  cfgfile: null
  cstratfile: null
  hconffile: null

crop:
  nrow: 153 # Do not modify
  ncol: 153 # Do not modify

# model-specific settings that will be used to initialize the imager (cgi-howfsc for compact model and corgihowfsc for full model).
models:
  cgi-howfsc:
    backend_type: "cgi-howfsc"
    normalization_type: "eetc"
    estimator: "default" # options are "perfect" and "default"; perfect estimator uses the model e-field for estimation, while default estimator uses the standard pairwise probing method
    starting_contrast: 6e-10 # starting contrast value for dmstartmap_filenames; not actually used in compact model
    dmstartmap_filenames:
    # Initial DM shape. Provide either:
    # - two filenames relative to modelpath, or
    # - two absolute paths.
    # Mixed absolute/relative entries are not supported.
      - "gitl_start_compact_dm1.fits"  # can provide full path including filename or just a filename if the file is located in the modelpath
      - "gitl_start_compact_dm2.fits"  # can provide full path including filename or just a filename if the file is located in the modelpath
    lrow: 436 # Do not modify
    lcol: 436 # Do not modify

  corgihowfsc:
    backend_type: "corgihowfsc"
    normalization_type: "eetc" # other options are "corgisim-off-axis" and "corgisim-on-axis"
    estimator: "default" # options are "perfect" and "default"; perfect estimator uses the model e-field for estimation, while default estimator uses the standard pairwise probing method
    starting_contrast: 3.5e-4 # starting contrast value for dmstartmap_filenames; if changing starting DM maps update this value
    dmstartmap_filenames:
      - "gitl_start_compact_dm1.fits"  # can provide full path including filename or just a filename if the file is located in the modelpath
      - "gitl_start_compact_dm2.fits"  # can provide full path including filename or just a filename if the file is located in the modelpath
    lrow: 0 # Do not modify
    lcol: 0 # Do not modify
    corgi_overrides:
      # Parameters forwarded to GitlImage initialization
      is_noise_free: false
      oversampling_factor: 3 # always odd number
```

---

## Field descriptions

### `active_model`

Chooses which block under `models` is active.

| Value | Description |
| --- | --- |
| `cgi-howfsc` | Compact model — fastest option; both Jacobian and images from the compact model |
| `corgihowfsc` | Full `corgisim` model — slower but more realistic; images from `corgisim`, Jacobian still from compact model |

---

### `runtime`

Controls parallel execution.

| Field | Default | Description |
| --- | --- | --- |
| `num_proper_process` | `5` | Number of local PROPER processes used inside one image worker |
| `num_jac_process` | `6` | Number of local processes used for Jacobian computation when MPI is off |
| `num_imager_worker` | `null` | Number of outer image workers; `null` disables explicit outer parallelism; an integer caps active MPI worker ranks |
| `use_mpi` | `false` | Set `true` to enable MPI mode: rank 0 runs the main loop, all other ranks wait for work |

See [MPI and Multiprocessing](mpi_multiprocessing.md) for more detail.

---

### `sim_settings`

Controls the loop itself.

| Field | Default | Description |
| --- | --- | --- |
| `loop_framework` | `"corgi-howfsc"` | Do not modify |
| `precomp` | `"precomp_jacs_always"` | Jacobian loading/computation strategy (see below) |
| `output_every_iter` | `true` | `true`: save outputs every iteration; `false`: save only after the full run |
| `niter` | `3` | Number of loop iterations |
| `mode` | `"nfov_band1"` | Coronagraph mode; valid choices depend on the available model configuration files |
| `dark_hole` | `"360deg"` | Dark-hole geometry |
| `probe_shape` | `"default"` | Probe configuration |

**`precomp` values:**

| Value | Meaning |
| --- | --- |
| `precomp_jacs_always` | Recompute Jacobians during the loop as needed |
| `precomp_all_once` | Precompute the full set once before the loop starts |
| `load_all` | Load Jacobian products from disk; requires a valid `paths.defjacpath` |

---

### `paths`

Controls where outputs and Jacobian data are written.

| Field | Default | Description |
| --- | --- | --- |
| `base_path` | `"~"` | Root directory for all outputs |
| `base_corgiloop_path` | `"corgiloop_data"` | Output subdirectory under `base_path` |
| `defjacpath` | `"temp"` | Directory for Jacobian products; must point to an existing directory with the needed files when `precomp: load_all` |
| `final_filename` | `"final_frames.fits"` | Filename for the final saved frame FITS file |
| `folder_tag` | `null` | Optional string appended to the per-run output folder name |

---

### `path_overrides`

Overrides the mode-derived configuration file paths. Leave any field as `null` to use the default path selected from `mode`. When overriding, use absolute paths.

| Field | Default | Description |
| --- | --- | --- |
| `cfgfile` | `null` | Override for the optical configuration YAML |
| `cstratfile` | `null` | Override for the control-strategy YAML |
| `hconffile` | `null` | Override for the hardware configuration YAML |

---

### `crop`

Controls the output frame dimensions. Do not modify these values.

| Field | Default |
| --- | --- |
| `nrow` | `153` |
| `ncol` | `153` |

---

### `models`

Contains one block per supported image-generation model. Only the block named by `active_model` is used.

#### `models.cgi-howfsc` — compact model

| Field | Default | Description |
| --- | --- | --- |
| `backend_type` | `"cgi-howfsc"` | Do not modify |
| `normalization_type` | `"eetc"` | Contrast normalisation strategy; `eetc` uses the engineering exposure time calculator |
| `estimator` | `"default"` | `"default"`: pairwise probing estimator; `"perfect"`: uses model e-field directly |
| `starting_contrast` | `6e-10` | Not used in the compact model path |
| `dmstartmap_filenames` | *(two filenames)* | Initial DM1 and DM2 shapes; provide two filenames relative to `modelpath` or two absolute paths — do not mix |
| `lrow` | `436` | Do not modify |
| `lcol` | `436` | Do not modify |

#### `models.corgihowfsc` — full corgisim model

| Field | Default | Description |
| --- | --- | --- |
| `backend_type` | `"corgihowfsc"` | Do not modify |
| `normalization_type` | `"eetc"` | `"eetc"`: engineering ETC; `"corgisim-off-axis"`: off-axis source at 7λ/D; `"corgisim-on-axis"`: FPM removed from beam |
| `estimator` | `"default"` | `"default"`: pairwise probing estimator; `"perfect"`: uses model e-field directly |
| `starting_contrast` | `3.5e-4` | Used to set initial camera parameters; if wrong, early exposure times may be poor |
| `dmstartmap_filenames` | *(two filenames)* | Same rules as `cgi-howfsc` block above |
| `lrow` | `0` | Do not modify |
| `lcol` | `0` | Do not modify |

**`models.corgihowfsc.corgi_overrides`** — forwarded to `GitlImage` initialisation:

| Field | Default | Description |
| --- | --- | --- |
| `is_noise_free` | `false` | `true`: disable image noise (useful for debugging or controlled comparisons) |
| `oversampling_factor` | `3` | Must always be an odd number |

Any `proper` keyword can be added here as long as the key matches the relevant `proper_keyword`.

---

## Common mistakes

- Launching with `use_mpi: true` without `mpiexec`, `mpirun`, or `srun`
- Setting too many workers for the available CPUs
- Mixing relative and absolute DM-map paths in `dmstartmap_filenames`
- Setting `precomp: load_all` without providing the needed Jacobian files at `defjacpath`
- Changing fields marked "Do not modify" without checking downstream assumptions

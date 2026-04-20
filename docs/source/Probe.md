# Probe
## Focal plane wavefront sensing: pairwise probing (PWP)

### Theoretical framework

In the code, the focal plane wavefront sensing uses Pairwise Probing (PWP) to estimate the complex electric field of the coherent starlight speckles. By applying known phase perturbations ( = probes) to the DM, we modulate the coherent stellar flux to extract the speckles electric field.

## Software implementation 
With corgihowfsc, you can select from several types of probes as soon as you start your simulation. 
### Core sensing interface (`corgihowfsc/sensing/`)

#### `Probes.py`
This file defines the `Probes` class. It establishes the basic methods required for the simulation:
* `get_dm_probes()`: Returns the voltage {$V_{+}, V_{-}$} corresponding to the requested probe {$I_{+}, I_{-}$} 
* `get_probe_ap()`: Computes the analytical complex electric field ($\Delta p$) of the probe in the focal plane. To see the images, go find the result of your simulation to `corgiloop_data/corgi-howfsc_gitl`. For each iterations, you will find `images.fits` which contains, for each subband, one image without a probe and six with a probe.
```
├── config.yml
├── ...
├── iteration_0001
│   ├── dm1_command.fits
│   ├── dm2_command.fits
│   ├── efield_estimations.fits
│   ├── images.fits
│   ├── intensity_coherent.fits
│   ├── intensity_incoherent.fits
│   ├── intensity_total.fits
│   └── perfect_efields.fits
├── iteration_0002
│   └── ...
├── ...
└── iteration_XXXX
    └── ...
```
#### `GettingProbes.py`
This file defines the `ProbesShapes` class. It's a class that inherits from `Probes`. 
It's responsible for reading the pre-computed probe shapes. These are stored as `.fits` files in `corgihowfsc/model/probes"`.

### Generation scripts (`corgihowfsc/scripts/`)

The complex spatial geometries of the probes are synthesized offline before running a loop. These scripts generate the 2D voltage maps that will be loaded by `GettingProbes.py`.

#### `write_sinc_probes.py`
Generates the default "Sinc-Sinc-Sine" probing maps. This is the default shape on board

#### `write_gaussian_probes.py`
An alternative generator that produces Gaussian probes on the DM. They should be the preferred choice of alternative probes for the actual instrument

### Choosing your probe shape for a run (`corgihowfsc/scripts/default_param.yml"`)
Before running a loop, you can choose from 4 different probe shapes `{'default', 'single', 'gaussian', 'unmodulated_sinc'}` by specifying it in the YAML file under the section `sim_settings` using the parameter `probe_shape`:
```yaml
sim_settings:
  loop_framework: "corgi-howfsc" # do not modify 
  precomp: "precomp_jacs_always" # options: 'precomp_jacs_always', 'precomp_all_once', 'load_all' (only if defjacpath is not None)
  output_every_iter: true # whether to save the output frames at every iteration (true) or after all iterations are done (false)
  niter: 3 # number of iterations to run
  mode: "nfov_band1" # options: see models
  dark_hole: "360deg"  # options: see models
  probe_shape: "gaussian"
 ```
Please note: if you want to control the amplitude of your probe, you should be aware that the code automatically applies a multiplier factor `scale_factor_list` depending, in particular, on the current contrast level.